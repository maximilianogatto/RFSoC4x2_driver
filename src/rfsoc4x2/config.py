"""The static description of a setup: which elements exist and which pulses.

Holds specs only, so a Config can be written, read and checked without a board.
"""

from dataclasses import asdict, dataclass, field
from typing import List

from .specs import ElementSpec, PulseSpec


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

    def to_dict(self) -> dict:
        """JSON-compatible copy of every spec, stored with each dataset by `RFSoC.run()`."""
        return {
            "elements": [asdict(element) for element in self.elements],
            "pulses": [asdict(pulse) for pulse in self.pulses],
        }

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
