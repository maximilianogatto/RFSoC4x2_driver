"""
Specifications to use qubit, resonator and pulses. Used by the config module to create elements and pulses from specs.

They don't need qick instrument to be created.
"""

from abc import ABC
from dataclasses import dataclass, field

# ---------------------- ELEMENT SPECS ----------------------

@dataclass(kw_only=True)
class ElementSpec(ABC):
    """Base description of an element (a logical channel of the setup)."""

    name: str
    dac: int
    frequency: float                  # Hz, carrier of every pulse of this element
    nqz: int = 1                      # Nyquist zone of the DAC


@dataclass(kw_only=True)
class QubitSpec(ElementSpec):
    """A qubit drive line. Output only, so it has no ADC."""


@dataclass(kw_only=True)
class ResonatorSpec(ElementSpec):
    """A readout resonator. Needs an ADC to listen to its own tone."""

    adc: int
    readout_length: float             # sec, ADC integration window
    time_of_flight: float = 0.0       # sec, DAC -> fridge -> amps -> ADC round trip
    #                                   Calibrate once per cooldown: the ADC window
    #                                   opens this long after the pulse is played.


# ---------------------- ENVELOPE SPECS ----------------------

@dataclass(kw_only=True)
class EnvelopeSpec(ABC):
    """Base description of a pulse envelope.

    An envelope is a waveform loaded into the DAC's envelope memory. It is
    referenced by `ArbitraryPulseSpec`.
    """

    name: str


@dataclass(kw_only=True)
class GaussianEnvelopeSpec(EnvelopeSpec):
    """Gaussian envelope. Maps to `qickodes.envelopes_v2.GaussianEnvelope`."""

    sigma: float                      # sec
    length: float                     # sec, total envelope length


@dataclass(kw_only=True)
class GaussianDragEnvelopeSpec(EnvelopeSpec):
    """Gaussian envelope with DRAG correction.

    Maps to `qickodes.envelopes_v2.GaussianDragEnvelope`. The DRAG term
    suppresses leakage to the second excited state, which a bare gaussian
    drives because its spectrum reaches the 1-2 transition.
    """

    sigma: float                      # sec
    length: float                     # sec, total envelope length
    delta: float                      # Hz, qubit anharmonicity
    alpha: float                      # dimensionless DRAG coefficient


# ---------------------- PULSE SPECS ----------------------

@dataclass(kw_only=True)
class PulseSpec(ABC):
    """Base description of a pulse.

    The DAC and the carrier frequency belong to the element that owns the
    pulse, not to the pulse itself: a pulse only knows which element it belongs
    to, and its detuning from that element's frequency.
    """

    name: str
    element: str                      # name of the ElementSpec that owns it
    detuning: float = 0.0             # Hz, offset from the element frequency
    phase: float = 0.0                # deg
    gain: float = 1.0                 # -1 .. 1
    reset_phase: bool = False
    hold_last_sample: bool = False
    periodic: bool = False            # play continuously instead of once


@dataclass(kw_only=True)
class ConstantPulseSpec(PulseSpec):
    """Rectangular pulse. Maps to `ConstantPulse`.

    A burst of a tone: constant amplitude for `length`, at the element's
    frequency plus `detuning`. This is the readout pulse.
    """

    length: float                     # sec


@dataclass(kw_only=True)
class ArbitraryPulseSpec(PulseSpec):
    """Pulse with an arbitrary envelope. Maps to `ArbitraryPulse`.

    There is no `length` field: the duration of an arbitrary pulse is the
    length of its envelope. This is the qubit drive pulse.
    """

    envelope: EnvelopeSpec
