"""Small sequence fragments: lists of macros you concatenate into a program."""

from qickodes.macro_base_v2 import Macro
from qickodes.macros_v2 import DelayAuto, PlayPulse, Trigger

from .elements import Element


def readout(element: Element, time_of_flight: float | None = None, t_delay: float = 10e-9) -> list[Macro]:
    """Return a readout sequence for a given element.

    The acquisition window opens `time_of_flight` after the pulse is played.
    Leave it as None to use the element's calibrated value.
    """

    if element.adc is None:
        raise ValueError(
            f"element '{element.name}' has no ADC, so it cannot be read out. "
            f"Only a resonator can."
        )

    if time_of_flight is None:
        time_of_flight = element.time_of_flight

    return [
        DelayAuto(element.qi, t=t_delay),
        Trigger(element.qi, element.adc, t=time_of_flight),
        PlayPulse(element.qi, element.pulse("readout")),
        DelayAuto(element.qi, t=t_delay),
    ]


def play(element: Element, pulse_name: str, t_delay: float = 10e-9) -> list[Macro]:
    """Return a play sequence for a given element and pulse"""

    return [
        DelayAuto(element.qi, t=t_delay),
        PlayPulse(element.qi, element.pulse(pulse_name)),
        DelayAuto(element.qi, t=t_delay),
    ]


def wait(element: Element, t_delay: float) -> list[Macro]:
    """Return a wait sequence for a given element"""

    return [
        DelayAuto(element.qi, t=t_delay),
    ]
