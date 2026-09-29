"""
Specifications to use qubit, resonator and pulses. Used by the config module to create elements and pulses from specs.

They don't need qick instrument to be created.
"""

from abc import ABC
from dataclasses import dataclass, field
from typing import Sequence

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
    """A qubit drive line. Output only, so it has no ADC.

    A qubit is read out through a resonator, so `readout` names the
    ResonatorSpec coupled to it. Measurements use it as their default readout,
    so you pass the qubit alone and the right resonator comes with it. It is a
    name rather than an object, like `PulseSpec.element`, so the spec stays
    plain data and saves to JSON unchanged.

    Leave it None for a qubit with no readout of its own; a program then needs
    the resonator passed explicitly. Two qubits may name the same resonator.
    """

    readout: str | None = None        # name of the ResonatorSpec that reads it out


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
    referenced by `ArbitraryPulseSpec` and `FlatTopPulseSpec`.
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

    Deliberately minimal: the DAC and the carrier frequency belong to the
    element that owns the pulse, so a pulse only knows which element it belongs
    to. `MuxedConstantPulseSpec` stops here, because a multiplexed pulse has no
    carrier of its own - the frequencies live on the DAC's tones.
    """

    name: str
    element: str                      # name of the ElementSpec that owns it


@dataclass(kw_only=True)
class TonePulseSpec(PulseSpec):
    """A pulse that carries its own tone. Everything except the muxed one.

    `periodic` is NOT here: FlatTopPulse has no periodic mode, so it sits on
    the concrete classes that actually support it.
    """

    detuning: float = 0.0             # Hz, offset from the element frequency
    phase: float = 0.0                # deg
    gain: float = 1.0                 # -1 .. 1
    reset_phase: bool = False
    hold_last_sample: bool = False


@dataclass(kw_only=True)
class ConstantPulseSpec(TonePulseSpec):
    """Rectangular pulse. Maps to `ConstantPulse`.

    A burst of a tone: constant amplitude for `length`, at the element's
    frequency plus `detuning`. This is the usual readout pulse.
    """

    length: float                     # sec
    periodic: bool = False            # play continuously instead of once


@dataclass(kw_only=True)
class CorrectedConstantPulseSpec(TonePulseSpec):
    """Rectangular pulse with a frequency-dependent gain and phase correction.

    Maps to `CorrectedConstantPulse`. The three arrays are interpolated at the
    pulse frequency to derive a gain factor and a phase offset, which is how
    you flatten the response of a cable and amplifier chain. Leave them empty
    for no correction.
    """

    length: float                     # sec
    periodic: bool = False
    correctable_freqs: Sequence[float] = field(default_factory=list)   # Hz
    gain_factors: Sequence[float] = field(default_factory=list)
    phase_offsets: Sequence[float] = field(default_factory=list)       # deg


@dataclass(kw_only=True)
class ArbitraryPulseSpec(TonePulseSpec):
    """Pulse with an arbitrary envelope. Maps to `ArbitraryPulse`.

    There is no `length` field: the duration of an arbitrary pulse is the
    length of its envelope. This is the usual qubit drive pulse.
    """

    envelope: EnvelopeSpec
    periodic: bool = False


@dataclass(kw_only=True)
class FlatTopPulseSpec(TonePulseSpec):
    """Flat-top pulse with arbitrary ramps. Maps to `FlatTopPulse`.

    `length` is the FLAT portion only; the ramps come from the envelope, whose
    first half is the ramp-up and second half the ramp-down. Use an even-length
    envelope. FlatTopPulse has no periodic mode.
    """

    envelope: EnvelopeSpec
    length: float                     # sec, flat portion only


@dataclass(kw_only=True)
class MuxedConstantPulseSpec(PulseSpec):
    """Frequency-multiplexed rectangular pulse. Maps to `MuxedConstantPulse`.

    Several tones at once on one DAC, which is how you read out several
    resonators through one feedline. Frequency and gain are properties of the
    DAC's tones, not of this pulse; here you only choose which tones to play.

    Needs a MultiplexedDacChannel, so the firmware must be built with a muxed
    generator. The standard RFSoC 4x2 image is not.
    """

    length: float                     # sec
    tone_nums: Sequence[int] = field(default_factory=tuple)
