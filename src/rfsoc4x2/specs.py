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
    type: str = field(default="", init=False)


@dataclass(kw_only=True)
class QubitSpec(ElementSpec):
    """A qubit drive line. Output only, so it has no ADC."""

    type: str = field(default="qubit", init=False)


@dataclass(kw_only=True)
class ResonatorSpec(ElementSpec):
    """A readout resonator. Needs an ADC to listen to its own tone."""

    type: str = field(default="resonator", init=False)
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
    type: str = field(default="", init=False)


@dataclass(kw_only=True)
class GaussianEnvelopeSpec(EnvelopeSpec):
    """Gaussian envelope. Maps to `qickodes.envelopes_v2.GaussianEnvelope`."""

    type: str = field(default="gaussian", init=False)
    sigma: float                      # sec
    length: float                     # sec, total envelope length


@dataclass(kw_only=True)
class GaussianDragEnvelopeSpec(EnvelopeSpec):
    """Gaussian envelope with DRAG correction.

    Maps to `qickodes.envelopes_v2.GaussianDragEnvelope`.
    """

    type: str = field(default="gaussian_drag", init=False)
    sigma: float                      # sec
    length: float                     # sec, total envelope length
    delta: float                      # Hz, qubit anharmonicity
    alpha: float                      # dimensionless DRAG coefficient


# ---------------------- PULSE SPECS ----------------------

@dataclass(kw_only=True)
class PulseSpec(ABC):
    """Base description of a pulse.

    Deliberately minimal: the DAC and the carrier frequency belong to the
    element that owns the pulse, not to the pulse itself. A pulse only knows
    which element it belongs to, and its detuning from that element's
    frequency.
    """

    name: str
    element: str                      # name of the ElementSpec that owns it
    type: str = field(default="", init=False)


@dataclass(kw_only=True)
class _TonePulseSpec(PulseSpec):
    """Fields shared by every pulse that carries its own tone.

    All pulses except the multiplexed one, whose frequency and gain live on
    the DAC's tones rather than on the pulse.
    """

    detuning: float = 0.0             # Hz, offset from the element frequency
    phase: float = 0.0                # deg
    gain: float = 1.0                 # -1 .. 1
    reset_phase: bool = False
    hold_last_sample: bool = False


@dataclass(kw_only=True)
class ConstantPulseSpec(_TonePulseSpec):
    """Rectangular pulse. Maps to `ConstantPulse`."""

    type: str = field(default="constant", init=False)
    length: float                     # sec
    periodic: bool = False


@dataclass(kw_only=True)
class CorrectedConstantPulseSpec(_TonePulseSpec):
    """Rectangular pulse with a frequency-dependent gain/phase correction.

    Maps to `CorrectedConstantPulse`. The three arrays are interpolated at the
    pulse frequency to derive a gain factor and a phase offset. Leave them
    empty for no correction.
    """

    type: str = field(default="corrected_constant", init=False)
    length: float                     # sec
    periodic: bool = False
    correctable_freqs: Sequence[float] = field(default_factory=list)   # Hz
    gain_factors: Sequence[float] = field(default_factory=list)
    phase_offsets: Sequence[float] = field(default_factory=list)       # deg


@dataclass(kw_only=True)
class ArbitraryPulseSpec(_TonePulseSpec):
    """Pulse with an arbitrary envelope. Maps to `ArbitraryPulse`.

    There is no `length` field: the duration of an arbitrary pulse is the
    length of its envelope.
    """

    type: str = field(default="arbitrary", init=False)
    envelope: EnvelopeSpec
    periodic: bool = False


@dataclass(kw_only=True)
class FlatTopPulseSpec(_TonePulseSpec):
    """Flat-top pulse with arbitrary ramps. Maps to `FlatTopPulse`.

    `length` is the FLAT portion only; the ramps come from the envelope, whose
    first half is the ramp-up and second half the ramp-down. Use an
    even-length envelope.

    `FlatTopPulse` has no periodic mode, so there is no `periodic` field.
    """

    type: str = field(default="flat_top", init=False)
    envelope: EnvelopeSpec
    length: float                     # sec, flat portion only


@dataclass(kw_only=True)
class MuxedConstantPulseSpec(PulseSpec):
    """Frequency-multiplexed rectangular pulse. Maps to `MuxedConstantPulse`.

    Requires a `MultiplexedDacChannel`, which only exists if the firmware was
    built with a muxed generator. Frequency and gain are properties of the
    DAC's tones, not of this pulse; here you only choose which tones to play.
    """

    type: str = field(default="muxed_constant", init=False)
    length: float                     # sec
    tone_nums: Sequence[int] = field(default_factory=tuple)
