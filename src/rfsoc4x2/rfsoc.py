"""The live board: holds the connection, the elements and their pulses."""
from collections.abc import Sequence
from pathlib import Path
from dataclasses import dataclass, field, fields
from typing import Callable, Literal, Optional

from qcodes import Instrument, Station, Measurement, initialise_or_create_database_at, load_or_create_experiment

from qickodes.instrument_v2 import QickInstrument, SoftwareSweep
from qickodes.pulse_base_v2 import DacPulse
from qickodes.macro_base_v2 import Macro

from .build import build_element, build_pulse
from .config import Config
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
    acquisition_mode: Literal[
        "accumulated",
        "accumulated geometric median",
        "accumulated shots",
        "ddr4",
        "decimated",
        "state population",
    ] = "accumulated"
    
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
        self.pulses: dict[str, DacPulse] = {}

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
            self.pulses = {}

    def element(self, name: str) -> Element:
        """Return one element by name, with a clear error if it is not there."""
        if name not in self.elements:
            raise KeyError(f"no element '{name}'. Configured elements: {sorted(self.elements)}")
        return self.elements[name]

    def pulse(self, name: str) -> DacPulse:
        """Return one pulse by name, with a clear error if it is not there."""
        if name not in self.pulses:
            raise KeyError(f"no pulse '{name}'. Configured pulses: {sorted(self.pulses)}")
        return self.pulses[name]
    
    def add_element(self, element_spec: ElementSpec):
        """Add a new element to the RFSoC and apply its spec to the hardware."""
        if element_spec.name in self.elements:
            raise ValueError(f"element '{element_spec.name}' already exists. Use update_spec() to change it.")
        element = build_element(element_spec, self.qi)
        self.elements[element.name] = element
        print(f"Added new element {element.name} ({element.type}) on DAC {element_spec.dac}.")
    
    def add_pulse(self, pulse_spec: PulseSpec):
        """Add a new pulse to an existing element and apply its spec to the hardware."""
        if pulse_spec.name in self.pulses:
            raise ValueError(f"pulse '{pulse_spec.name}' already exists. Use update_spec() to change it.")
        element = self.element(pulse_spec.element)
        pulse = build_pulse(pulse_spec, element.dac, name=pulse_spec.name, base_freq=element.frequency)
        element.pulses[pulse_spec.name] = pulse
        self.pulses[pulse_spec.name] = pulse
        print(f"Added new pulse {pulse_spec.name} ({pulse_spec.type}) on {element.name}.")

    def _apply_config(self):
        """Build every element, then every pulse, from the config."""
        for element_spec in self.config.elements:
            element = build_element(element_spec, self.qi)
            self.elements[element.name] = element
            print(f"  element {element.name} ({element.type}) on DAC {element_spec.dac}")

        for pulse_spec in self.config.pulses:
            element = self.element(pulse_spec.element)
            pulse = build_pulse(pulse_spec, element.dac, name=pulse_spec.name, base_freq=element.frequency)

            element.pulses[pulse_spec.name] = pulse
            self.pulses[pulse_spec.name] = pulse
            print(f"  pulse {pulse_spec.name} ({pulse_spec.type}) on {element.name}")
            
    def _pulse_spec(self, name: str):
        """The spec a pulse was built from."""
        for spec in self.config.pulses:
            if spec.name == name:
                return spec
        raise KeyError(f"no pulse spec '{name}'. Configured: {self.config.pulse_names()}")

    def update_spec(self, parameters: list[tuple[str, float]], element: str | None = None, pulse: str | None = None, verbose: bool = True):
        """Update the spec of one element or pulse, and apply it to the hardware.

        The spec is the same object that lives in `Config`, so the new value is
        also what `run()` writes into the snapshot of every later dataset. This
        is how a calibration survives.

        Args:
            parameters: list of (parameter name, value) tuples to update
            element: name of the element to update (if None, update a pulse)
            pulse: name of the pulse to update (if None, update an element)
        """
        if (element is None) == (pulse is None):
            raise ValueError("Specify either an element or a pulse, not both.")

        if element is not None:
            spec, label = self.element(element).spec, element
        else:
            spec, label = self._pulse_spec(pulse), pulse

        if spec is None:
            raise ValueError(f"{label} was built without a spec, there is nothing to update.")

        for param_name, value in parameters:
            if not hasattr(spec, param_name):
                raise AttributeError(f"{label} has no spec parameter '{param_name}'")
            if verbose:
                print(f"Updating {label} spec: {param_name} from {getattr(spec, param_name)} to {value}")
            setattr(spec, param_name, value)

        # Push onto the objects that already exist. Do NOT rebuild: qickodes
        # pulses are qcodes InstrumentChannels, so building a second one with
        # the same name on the same DAC raises.
        if element is not None:
            self._resync_element(self.element(element))
        else:
            self._resync_pulse(pulse)
        print(f"Applied updated spec for {label} to hardware.")

    def _resync_element(self, element: Element):
        """Re-apply an element's spec to its existing channels and pulses."""
        spec = element.spec
        element.frequency = spec.frequency
        element.dac.nqz.set(spec.nqz)

        if element.adc is not None:
            element.adc.freq.set(spec.frequency)        # the ADC follows the resonator
            element.adc.length.set(spec.readout_length)
            element.time_of_flight = spec.time_of_flight

        for pulse_spec in self.config.pulses:           # pulses follow the carrier
            if pulse_spec.element == element.name and hasattr(pulse_spec, "detuning"):
                element.pulse(pulse_spec.name).freq.set(spec.frequency + pulse_spec.detuning)

    def _resync_pulse(self, name: str):
        """Re-apply a pulse's spec to the qickodes pulse that already exists.

        Envelope parameters (sigma, alpha, ...) live on a separate envelope
        object and are not resynced here: changing a pulse shape still needs a
        fresh RFSoC.
        """
        spec = self._pulse_spec(name)
        live = self.pulse(name)
        element = self.element(spec.element)

        for attr in ("phase", "gain", "length", "periodic",
                     "reset_phase", "hold_last_sample", "tone_nums"):
            if hasattr(spec, attr) and hasattr(live, attr):
                getattr(live, attr).set(getattr(spec, attr))

        if hasattr(spec, "detuning"):
            live.freq.set(element.frequency + spec.detuning)

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