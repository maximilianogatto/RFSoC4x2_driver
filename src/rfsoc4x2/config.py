"""The static description of a setup: which elements exist and which pulses.

Holds specs only, so a Config can be written, read and checked without a board.
"""

import inspect
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import List

from . import specs as _specs
from .specs import ElementSpec, EnvelopeSpec, PulseSpec

# Every spec class by name, discovered from the specs module so a new type is
# supported without editing a list here.
SPEC_CLASSES = {
    name: cls
    for name, cls in inspect.getmembers(_specs, inspect.isclass)
    if is_dataclass(cls) and issubclass(cls, (ElementSpec, PulseSpec, EnvelopeSpec))
}

# The values a measurement CALIBRATES, as opposed to how the setup is wired.
# Only these are written by save_calibration and applied by load_calibration.
CALIBRATED = (
    "frequency", "time_of_flight", "readout_length",   # element
    "gain", "phase", "detuning", "length",             # pulse
    "sigma", "alpha", "delta",                         # envelope
)


def _to_dict(spec) -> dict:
    """A spec as a dict, tagged with its class so it can be rebuilt."""
    out = {"class": type(spec).__name__}
    for f in fields(spec):
        value = getattr(spec, f.name)
        out[f.name] = _to_dict(value) if is_dataclass(value) else value
    return out


def _from_dict(data: dict):
    """Rebuild a spec from a dict written by `_to_dict`."""
    data = dict(data)
    name = data.pop("class", None)
    if name is None:
        raise ValueError(f"spec dict has no 'class' key: {sorted(data)}")
    if name not in SPEC_CLASSES:
        raise ValueError(f"unknown spec class '{name}'. Known: {sorted(SPEC_CLASSES)}")
    cls = SPEC_CLASSES[name]
    kwargs = {
        key: _from_dict(value) if isinstance(value, dict) and "class" in value else value
        for key, value in data.items()
    }
    return cls(**kwargs)


@dataclass
class Config:
    """Elements and pulses of one setup.

    Every pulse names the element it belongs to, and takes its DAC and its
    carrier frequency from that element.
    """

    elements: List[ElementSpec] = field(default_factory=list)
    pulses: List[PulseSpec] = field(default_factory=list)

    def __post_init__(self):
        self._validate_elements()
        self._validate_pulses()

    def element_names(self) -> list[str]:
        """Names of every element, in config order."""
        return [element.name for element in self.elements]

    def pulse_names(self) -> list[str]:
        """Names of every pulse, in config order."""
        return [pulse.name for pulse in self.pulses]

    def check_new_element(self, element: ElementSpec):
        """Raise if this element could not be added. Does not modify anything.

        Called before the element is built, so a bad spec fails before any
        hardware is touched.
        """
        if not isinstance(element, ElementSpec):
            raise ValueError(f"Element {element} is not an ElementSpec instance.")
        if element.name in self.element_names():
            raise ValueError(
                f"element '{element.name}' already exists. Use update_spec() to change it."
            )

    def check_new_pulse(self, pulse: PulseSpec):
        """Raise if this pulse could not be added. Does not modify anything."""
        if not isinstance(pulse, PulseSpec):
            raise ValueError(f"Pulse {pulse} is not a PulseSpec instance.")
        if pulse.name in self.pulse_names():
            raise ValueError(
                f"pulse '{pulse.name}' already exists. Use update_spec() to change it."
            )
        if pulse.element not in self.element_names():
            raise ValueError(
                f"Pulse {pulse.name} belongs to element '{pulse.element}', "
                f"which is not in the config. Known elements: {self.element_names()}"
            )

    # ---------------- the whole setup ----------------

    def to_dict(self) -> dict:
        """JSON-compatible copy of every spec, stored with each dataset by `RFSoC.run()`.

        Each spec carries its class name, so `from_dict` can rebuild it. The
        name is written here rather than stored on the specs, so there is no
        field that can fall out of step with the class.
        """
        return {
            "elements": [_to_dict(element) for element in self.elements],
            "pulses": [_to_dict(pulse) for pulse in self.pulses],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Rebuild a Config from `to_dict` output."""
        return cls(
            elements=[_from_dict(d) for d in data.get("elements", [])],
            pulses=[_from_dict(d) for d in data.get("pulses", [])],
        )

    def save(self, path) -> Path:
        """Write the whole setup to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        print(f"Saved setup to {path}.")
        return path

    @classmethod
    def load(cls, path) -> "Config":
        """Read a whole setup back from a JSON file."""
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_run(cls, run_id: int) -> "Config":
        """The setup that produced a dataset, read back out of its snapshot.

        `RFSoC.run()` puts the config into the station metadata of every run,
        so an old measurement can be reproduced exactly.
        """
        from qcodes.dataset import load_by_id

        snapshot = load_by_id(run_id).snapshot or {}
        stored = snapshot.get("station", {}).get("metadata", {}).get("rfsoc4x2_config")
        if stored is None:
            raise KeyError(
                f"run {run_id} has no 'rfsoc4x2_config' in its snapshot. It was "
                f"probably not taken with RFSoC.run()."
            )
        return cls.from_dict(stored)

    # ---------------- just the measured numbers ----------------

    def calibration(self) -> dict:
        """Only the values a measurement calibrates, keyed by spec name.

        Wiring - dac, adc, nqz, which pulses exist - is deliberately left out,
        so this can be laid over a different setup.
        """
        out = {"elements": {}, "pulses": {}}
        for element in self.elements:
            out["elements"][element.name] = {
                f.name: getattr(element, f.name)
                for f in fields(element) if f.name in CALIBRATED
            }
        for pulse in self.pulses:
            values = {
                f.name: getattr(pulse, f.name)
                for f in fields(pulse) if f.name in CALIBRATED
            }
            envelope = getattr(pulse, "envelope", None)
            if envelope is not None:
                values.update({
                    f.name: getattr(envelope, f.name)
                    for f in fields(envelope) if f.name in CALIBRATED
                })
            out["pulses"][pulse.name] = values
        return out

    def _validate_elements(self):
        """Check that elements are specs and that their names are unique."""
        seen = []
        for element in self.elements:
            if not isinstance(element, ElementSpec):
                raise ValueError(f"Element {element} is not an ElementSpec instance.")
            if element.name in seen:
                raise ValueError(f"Duplicated element name: {element.name}")
            seen.append(element.name)

    def _validate_pulses(self):
        """Check that pulses are specs, unique, and point at a known element."""
        known = self.element_names()
        seen = []
        for pulse in self.pulses:
            if not isinstance(pulse, PulseSpec):
                raise ValueError(f"Pulse {pulse} is not a PulseSpec instance.")
            if pulse.name in seen:
                raise ValueError(f"Duplicated pulse name: {pulse.name}")
            if pulse.element not in known:
                raise ValueError(
                    f"Pulse {pulse.name} belongs to element '{pulse.element}', "
                    f"which is not in the config. Known elements: {known}"
                )
            seen.append(pulse.name)
