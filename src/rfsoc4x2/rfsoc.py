"""The live board: holds the connection, the elements and their pulses."""
import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from dataclasses import dataclass, field, fields
from typing import Callable, Optional

from qcodes import Instrument, Station, Measurement, initialise_or_create_database_at, load_or_create_experiment

from qickodes.instrument_v2 import QickInstrument, SoftwareSweep
from qickodes.macro_base_v2 import Macro

from .build import build_element, build_pulse
from .config import Config, check_attenuation
from .elements import Element
from .specs import ElementSpec, PulseSpec

# The modes qickodes accepts. "raw" is not one of them.
ACQUISITION_MODES = (
    "accumulated",
    "accumulated geometric median",
    "accumulated shots",
    "ddr4",
    "decimated",
    "state population",
)

# These keep every shot, so qickodes forbids software averaging on top.
SHOT_RESOLVED_MODES = (
    "accumulated geometric median",
    "accumulated shots",
    "ddr4",
    "state population",
)


@dataclass
class RunConfig:
    """Everything one run needs that is not the pulse sequence itself.

    These settings belong to the run, not to the elements: averaging, sweeps,
    acquisition mode, and where the dataset is filed.
    """

    measurement_name: str
    experiment_name: str
    sample_name: str

    # averaging
    n_shots: int = 1000                  # qi.hard_avgs: repeated and averaged on the FPGA
    soft_avgs: int = 1                   # whole program re-run and averaged in python
    final_delay: Optional[float] = None  # sec between shots; None leaves qi untouched

    # sweeps
    software_sweeps: Sequence[SoftwareSweep] = field(default_factory=tuple)
    hardware_loop_counts: dict[str, int] = field(default_factory=dict)

    # acquisition
    acquisition_mode: str = "accumulated"   # one of ACQUISITION_MODES
    
    num_states: int = 0
    state_classifier: Optional[Callable] = None
    save_shots_as_npy: bool = False

    def __post_init__(self):
        self._validate()

    def _validate(self):
        """Fail here with a sentence, instead of deep inside qickodes."""
        if self.acquisition_mode not in ACQUISITION_MODES:
            raise ValueError(
                f"unknown acquisition_mode '{self.acquisition_mode}'. "
                f"Valid modes: {list(ACQUISITION_MODES)}"
            )

        if self.n_shots < 1:
            raise ValueError(f"n_shots must be at least 1, got {self.n_shots}")

        if self.soft_avgs < 1:
            raise ValueError(f"soft_avgs must be at least 1, got {self.soft_avgs}")

        if self.acquisition_mode in SHOT_RESOLVED_MODES and self.soft_avgs != 1:
            raise ValueError(
                f"acquisition_mode '{self.acquisition_mode}' keeps every shot, "
                f"so soft_avgs must be 1, got {self.soft_avgs}"
            )

        if self.acquisition_mode == "state population":
            if self.num_states < 2:
                raise ValueError(f"'state population' needs num_states >= 2, got {self.num_states}")
            if self.state_classifier is None:
                raise ValueError("'state population' needs a state_classifier")

        if self.final_delay is not None and self.final_delay < 0:
            raise ValueError(f"final_delay must be >= 0, got {self.final_delay}")

    def to_dict(self) -> dict:
        """JSON-compatible run settings, stored with each dataset by `RFSoC.run()`.

        Sweeps and the state classifier are live objects and are left out; the swept
        values are stored in the dataset as setpoints anyway.
        """
        skip = ("software_sweeps", "state_classifier")
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name not in skip}


class RFSoC:
    def __init__(
        self,
        name: str,
        ip: str,
        station: Station,
        db_path: str | Path,
        config: Config,
        ns_port: int = 8888,
    ):
        # NOTE: qickodes looks for the Pyro proxy named "myqick", so the server
        # on the board must register under that name (pyro_service.py).
        self.station = station
        self.config = config
        self.elements: dict[str, Element] = {}

        # configuring database path for qcodes measurements, before connecting, so a bad
        # path fails without leaving a half-built instrument. sqlite does not create folders.
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Using database at {db_path}.")
        initialise_or_create_database_at(db_path)

        if Instrument.exist(name):
            print(f"Instrument {name} already exists. Closing existing instrument.")
            Instrument.find_instrument(name).close()
            print(f"Closed existing instrument {name}.")

        print(f"Creating new QickInstrument with name {name} and IP {ip}.")
        self.qi = QickInstrument(ip, name=name, ns_port=ns_port)

        print(f"Adding QickInstrument {name} to the station.")
        station.add_component(self.qi)
        print(f"QickInstrument {name} added to the station.")

        print(f"Applying configuration to QickInstrument {name}.")
        self._apply_config()
        print(f"Configuration applied to QickInstrument {name}.")

    def close(self):
        """Close the QickInstrument and remove it from the station."""
        if self.qi is not None:
            print(f"Closing QickInstrument {self.qi.name}.")
            name = self.qi.name
            self.qi.close()
            print(f"QickInstrument {name} closed.")
            self.station.remove_component(name)
            print(f"QickInstrument {name} removed from the station.")
            self.qi = None
            self.elements = {}
    
    def display(self, message: str):
        """Display a message on the board's front panel."""
        self.qi.soc.oled_write(message)
        
    def display_ready(self):
        """Display a ready message on the board's front panel."""
        self.qi.soc.oled_ready()

    def element(self, name: str) -> Element:
        """Return one element by name, with a clear error if it is not there."""
        if name not in self.elements:
            raise KeyError(f"no element '{name}'. Configured elements: {sorted(self.elements)}")
        return self.elements[name]

    def add_element(self, element_spec: ElementSpec):
        """Add a new element to the RFSoC and apply its spec to the hardware.

        The spec also joins the Config, so it reaches the dataset snapshot and
        update_spec() can find it later.
        """
        # validate before building, so a bad spec touches no hardware
        self.config.check_new_element(element_spec)
        element = build_element(element_spec, self.qi)
        self.elements[element.name] = element
        self._link_readout(element)
        self.config.elements.append(element_spec)
        print(f"Added new element {element.name} ({element.type}) on DAC {element_spec.dac}.")

    def add_pulse(self, pulse_spec: PulseSpec):
        """Add a new pulse to an existing element and apply its spec to the hardware.

        The spec also joins the Config, so it reaches the dataset snapshot, and
        update_spec() retunes it when its element's frequency changes.
        """
        self.config.check_new_pulse(pulse_spec)
        element = self.element(pulse_spec.element)
        pulse = build_pulse(pulse_spec, element.dac, name=pulse_spec.name, base_freq=element.frequency)
        element.pulses[pulse_spec.name] = pulse
        self.config.pulses.append(pulse_spec)
        print(f"Added new pulse {pulse_spec.name} ({type(pulse_spec).__name__}) on {element.name}.")

    def _link_readout(self, element: Element):
        """Point a qubit element at the resonator element its spec names."""
        name = getattr(element.spec, "readout", None)
        element.readout = self.elements[name] if name is not None else None

    def _apply_config(self):
        """Build every element, then every pulse, from the config."""
        for element_spec in self.config.elements:
            element = build_element(element_spec, self.qi)
            self.elements[element.name] = element
            print(f"  element {element.name} ({element.type}) on DAC {element_spec.dac}")

        # a second pass: a qubit may be listed before the resonator it names
        for element in self.elements.values():
            self._link_readout(element)
            if element.readout is not None:
                print(f"  {element.name} is read out by {element.readout.name}")

        for pulse_spec in self.config.pulses:
            element = self.element(pulse_spec.element)
            pulse = build_pulse(pulse_spec, element.dac, name=pulse_spec.name, base_freq=element.frequency)

            element.pulses[pulse_spec.name] = pulse
            print(f"  pulse {pulse_spec.name} ({type(pulse_spec).__name__}) on {element.name}")
            
    def update_spec(self, parameters: list[tuple[str, float]], element: str | None = None, pulse: str | None = None, verbose: bool = True):
        """Update a spec and apply it to the hardware. This is how a calibration sticks.

        The spec is the same object that lives in `Config`, so the new value is
        also what `run()` writes into the snapshot of every later dataset.

        Calibrated numbers live at three levels, and this reaches all of them:

            element   frequency, time_of_flight, readout_length
            pulse     gain, detuning, phase, length
            envelope  sigma, alpha, delta   (via the pulse that owns it)

        An attribute that is not on the pulse spec is looked for on its
        envelope, so a DRAG calibration writes to the pulse it belongs to:

            rfsoc.update_spec([("gain", pi_gain)], pulse="pi")
            rfsoc.update_spec([("alpha", best_alpha)], pulse="pi")

        Args:
            parameters: list of (parameter name, value) tuples to update
            element: name of the element to update, or
            pulse: name of the pulse to update. Exactly one of the two.
            verbose: print each change
        """
        if (element is None) == (pulse is None):
            raise ValueError("Specify either an element or a pulse, not both.")

        label = element if element is not None else pulse
        spec = self.element(element).spec if element is not None else self._pulse_spec(pulse)
        if spec is None:
            raise ValueError(f"{label} was built without a spec, there is nothing to update.")

        for param_name, value in parameters:
            # a pulse parameter that is not on the pulse belongs to its envelope
            target_spec = spec
            if not hasattr(spec, param_name):
                envelope = getattr(spec, "envelope", None)
                if envelope is not None and hasattr(envelope, param_name):
                    target_spec = envelope
                else:
                    raise AttributeError(f"{label} has no spec parameter '{param_name}'")
            if param_name == "readout":
                self.config.check_readout(label, value)
            if param_name.endswith("_attenuation"):
                check_attenuation(label, param_name, value)
            if verbose:
                where = "" if target_spec is spec else " (envelope)"
                print(f"Updating {label}{where}: {param_name} from {getattr(target_spec, param_name)} to {value}")
            setattr(target_spec, param_name, value)

        # Push onto the objects that already exist. Do NOT rebuild: qickodes
        # pulses are qcodes InstrumentChannels, so building a second one with
        # the same name on the same DAC raises.
        if element is not None:
            self._apply_element_spec(self.element(element))
        else:
            self._apply_pulse_spec(pulse)
        if verbose:
            print(f"Applied updated spec for {label} to hardware.")

    def _pulse_spec(self, name: str):
        """The spec a pulse was built from."""
        for spec in self.config.pulses:
            if spec.name == name:
                return spec
        raise KeyError(f"no pulse spec '{name}'. Configured: {self.config.pulse_names()}")

    def _apply_element_spec(self, target: Element):
        """Re-apply an element's spec to its existing channels and pulses."""
        spec = target.spec
        target.frequency = spec.frequency
        target.dac.nqz.set(spec.nqz)
        self._link_readout(target)

        if target.adc is not None:
            target.adc.freq.set(spec.frequency)        # the ADC follows the resonator
            target.adc.length.set(spec.readout_length)
            target.time_of_flight = spec.time_of_flight

        for pulse_spec in self.config.pulses:          # pulses follow the carrier
            if pulse_spec.element == target.name and hasattr(pulse_spec, "detuning"):
                target.pulse(pulse_spec.name).freq.set(spec.frequency + pulse_spec.detuning)

    def _apply_pulse_spec(self, name: str):
        """Re-apply a pulse's spec, and its envelope's, to the live objects.

        Envelope values reach the board on the next run: `_initialize()` calls
        add_gauss() from the envelope's current parameters every time the
        program is compiled.
        """
        spec = self._pulse_spec(name)
        element = self.element(spec.element)
        live = element.pulse(name)

        for attr in ("phase", "gain", "length", "periodic",
                     "reset_phase", "hold_last_sample", "tone_nums"):
            if hasattr(spec, attr) and hasattr(live, attr):
                getattr(live, attr).set(getattr(spec, attr))

        if hasattr(spec, "detuning"):
            live.freq.set(element.frequency + spec.detuning)

        envelope_spec = getattr(spec, "envelope", None)
        live_envelope = getattr(live, "envelope", None)
        if envelope_spec is not None and live_envelope is not None:
            for attr in ("sigma", "length", "delta", "alpha"):
                if hasattr(envelope_spec, attr) and hasattr(live_envelope, attr):
                    getattr(live_envelope, attr).set(getattr(envelope_spec, attr))

    # ---------------- calibration files ----------------

    def save_calibration(self, path=None) -> Path:
        """Write today's calibrated numbers to a JSON file.

        Only what a measurement calibrates, not the wiring, so the file can be
        laid over a different setup. Defaults to `calibration/cal_<date>.json`
        beside the database: a directory of dated files is also the record of
        how the sample drifted through the cooldown.
        """
        if path is None:
            path = Path("calibration") / f"cal_{date.today().isoformat()}.json"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.config.calibration(), indent=2))
        print(f"Saved calibration to {path}.")
        return path

    def load_calibration(self, path, strict: bool = True, verbose: bool = False):
        """Apply a saved calibration to the current setup.

        Every value goes through `update_spec`, so it takes the same path a
        live calibration does: the Config is updated, the change reaches the
        hardware, and it lands in the snapshot of every later run.

        Args:
            path: a file written by `save_calibration`.
            strict: raise if the file names an element or pulse this setup does
                not have. With False they are skipped and reported. Leave it
                True unless you know why: a silent skip means measuring with a
                stale value and not knowing.
            verbose: print every parameter as it is applied.
        """
        data = json.loads(Path(path).read_text())
        applied = skipped = 0

        for kind, names in (("elements", self.config.element_names()),
                            ("pulses", self.config.pulse_names())):
            for name, values in data.get(kind, {}).items():
                if name not in names:
                    message = f"{path}: no {kind[:-1]} '{name}' in this setup"
                    if strict:
                        raise KeyError(f"{message}. Pass strict=False to skip it.")
                    print(f"  skipped: {message}")
                    skipped += len(values)
                    continue
                target = {"elements": {"element": name}, "pulses": {"pulse": name}}[kind]
                self.update_spec(list(values.items()), verbose=verbose, **target)
                applied += len(values)

        print(f"Applied {applied} calibrated values from {path}"
              + (f", skipped {skipped}." if skipped else "."))

    def run(self, macros: Sequence[Macro], config: RunConfig) -> int:
        """Play one sequence and store the result. Returns the qcodes run id."""

        if not macros:
            raise ValueError("macros is empty, there is nothing to play.")

        # hardware configuration
        self.qi.hard_avgs.set(config.n_shots)
        self.qi.soft_avgs.set(config.soft_avgs)
        if config.final_delay is not None:
            self.qi.final_delay.set(config.final_delay)

        self.qi.set_macro_list(macros)

        # the setup and the run settings go into the station snapshot of the dataset
        self.station.metadata["rfsoc4x2_config"] = self.config.to_dict()
        self.station.metadata["rfsoc4x2_run"] = config.to_dict()

        # qcodes measurement
        experiment = load_or_create_experiment(config.experiment_name, config.sample_name)
        meas = Measurement(exp=experiment, station=self.station, name=config.measurement_name)

        # acquisition
        print(f"Running {config.measurement_name} ({config.acquisition_mode}).")
        run_id = self.qi.run(
            meas,
            software_sweeps=config.software_sweeps,
            hardware_loop_counts=config.hardware_loop_counts,
            acquisition_mode=config.acquisition_mode,
            num_states=config.num_states,
            state_classifier=config.state_classifier,
            save_shots_as_npy=config.save_shots_as_npy,
        )
        print(f"Stored as run {run_id}.")
        return run_id