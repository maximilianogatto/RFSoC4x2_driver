"""Builders that turn specs into live qickodes objects.

Each spec type gets its own builder, selected by `functools.singledispatch`.
Adding a new pulse type means adding one registered function here; nothing in
`RFSoC` changes.

Specs stay free of qickodes imports: all the binding happens in this module.
"""

from __future__ import annotations

from functools import singledispatch

import numpy as np

# NOTE: these must be real runtime imports, not `if TYPE_CHECKING` ones.
# `singledispatch.register` calls `typing.get_type_hints()` on each registered
# function, which resolves EVERY annotation in the signature - not just the
# first parameter it dispatches on. A name that only exists for the type
# checker raises NameError at import time.
from qickodes.channels_v2 import DacChannel, MultiplexedDacChannel
from qickodes.envelope_base_v2 import DacEnvelope
from qickodes.envelopes_v2 import GaussianDragEnvelope, GaussianEnvelope
from qickodes.instrument_v2 import QickInstrument
from qickodes.pulse_base_v2 import DacPulse
from qickodes.pulses_v2 import (
    ArbitraryPulse,
    ConstantPulse,
    CorrectedConstantPulse,
    FlatTopPulse,
    MuxedConstantPulse,
)

from .elements import Element
from .specs import (
    ArbitraryPulseSpec,
    ConstantPulseSpec,
    CorrectedConstantPulseSpec,
    ElementSpec,
    EnvelopeSpec,
    GaussianDragEnvelopeSpec,
    FlatTopPulseSpec,
    GaussianEnvelopeSpec,
    MuxedConstantPulseSpec,
    PulseSpec,
    TonePulseSpec,
    QubitSpec,
    ResonatorSpec,
)


def _known(dispatcher) -> list[str]:
    """Spec classes a dispatcher can build, for the error message."""
    return sorted(cls.__name__ for cls in dispatcher.registry if cls is not object)


# ---------------------- ELEMENTS ----------------------


@singledispatch
def build_element(spec: ElementSpec, qi: QickInstrument) -> Element:
    """Configure the channels of one element and return it.

    Parameters
    ----------
    spec : ElementSpec
        The element description. Its type selects the implementation.
    qi : QickInstrument
        The connected instrument whose channels are configured.
    """
    raise TypeError(
        f"no element builder for {type(spec).__name__}. Known: {_known(build_element)}"
    )


@build_element.register
def _(spec: QubitSpec, qi: QickInstrument) -> Element:
    # A qubit is drive only: one DAC, no ADC, nothing to match.
    dac = qi.dacs[spec.dac]
    dac.nqz.set(spec.nqz)
    return Element(
        name=spec.name,
        type="qubit",
        dac=dac,
        frequency=spec.frequency,
        adc=None,
        spec=spec,          # same object as in Config.elements, so updates persist
    )


@build_element.register
def _(spec: ResonatorSpec, qi: QickInstrument) -> Element:
    # A resonator drives a tone and listens to it, so the DAC and the ADC are
    # paired and tuned to the same frequency.
    dac = qi.dacs[spec.dac]
    dac.nqz.set(spec.nqz)

    adc = qi.adcs[spec.adc]
    adc.freq.set(spec.frequency)
    adc.length.set(spec.readout_length)

    dac.matching_adc.set(adc.channel_num)
    adc.matching_dac.set(dac.channel_num)

    return Element(
        name=spec.name,
        type="resonator",
        dac=dac,
        frequency=spec.frequency,
        adc=adc,
        time_of_flight=spec.time_of_flight,
        spec=spec,          # same object as in Config.elements, so updates persist
    )


# ---------------------- ENVELOPES ----------------------


@singledispatch
def build_envelope(spec: EnvelopeSpec, dac: DacChannel, name: str | None = None) -> DacEnvelope:
    """Create the qickodes envelope described by `spec` on `dac`.

    Called by the pulse builders; you rarely need it directly. The envelope
    lives in the DAC's own memory, which is why it must be built on the same
    DAC as the pulse that references it.
    """
    raise TypeError(
        f"no envelope builder for {type(spec).__name__}. Known: {_known(build_envelope)}"
    )


@build_envelope.register
def _(spec: GaussianEnvelopeSpec, dac: DacChannel, name: str | None = None) -> DacEnvelope:
    envelope = GaussianEnvelope(dac, name or spec.name)
    envelope.sigma.set(spec.sigma)
    envelope.length.set(spec.length)
    return envelope


@build_envelope.register
def _(spec: GaussianDragEnvelopeSpec, dac: DacChannel, name: str | None = None) -> DacEnvelope:
    envelope = GaussianDragEnvelope(dac, name or spec.name)
    envelope.sigma.set(spec.sigma)
    envelope.length.set(spec.length)
    envelope.delta.set(spec.delta)
    envelope.alpha.set(spec.alpha)
    return envelope


# ---------------------- PULSES ----------------------


def _apply_carrier(pulse, spec: TonePulseSpec, base_freq: float) -> None:
    """Set the parameters every pulse has in common.

    The pulse frequency is `base_freq + spec.detuning`, so moving an element's
    frequency moves all of its pulses. Call with `base_freq=0` to treat
    `detuning` as an absolute frequency.
    """
    pulse.freq.set(base_freq + spec.detuning)
    pulse.phase.set(spec.phase)
    pulse.gain.set(spec.gain)
    pulse.reset_phase.set(spec.reset_phase)
    pulse.hold_last_sample.set(spec.hold_last_sample)
    # periodic is deliberately not here: FlatTopPulse has no periodic mode


@singledispatch
def build_pulse(spec: PulseSpec, dac: DacChannel, name: str | None = None, base_freq: float = 0.0) -> DacPulse:
    """Create the qickodes pulse described by `spec` on `dac`.

    Parameters
    ----------
    spec : PulseSpec
        The pulse description. Its type selects the implementation.
    dac : DacChannel
        The DAC that will play the pulse.
    name : str, optional
        Name for the pulse, unique within the QickInstrument. Defaults to
        `spec.name`.
    base_freq : float, default=0.0
        Carrier frequency of the owning element, in Hz. The pulse is placed at
        `base_freq + spec.detuning`.
    """
    raise TypeError(
        f"no pulse builder for {type(spec).__name__}. Known: {_known(build_pulse)}"
    )


@build_pulse.register
def _(spec: ConstantPulseSpec, dac: DacChannel, name: str | None = None, base_freq: float = 0.0) -> DacPulse:
    pulse = ConstantPulse(dac, name or spec.name)
    _apply_carrier(pulse, spec, base_freq)
    pulse.length.set(spec.length)
    pulse.periodic.set(spec.periodic)
    return pulse


@build_pulse.register
def _(spec: ArbitraryPulseSpec, dac: DacChannel, name: str | None = None, base_freq: float = 0.0) -> DacPulse:
    pulse_name = name or spec.name
    envelope = build_envelope(spec.envelope, dac, f"{pulse_name}_envelope")
    # No length: an arbitrary pulse lasts as long as its envelope.
    pulse = ArbitraryPulse(dac, pulse_name, envelope)
    _apply_carrier(pulse, spec, base_freq)
    pulse.periodic.set(spec.periodic)
    return pulse


@build_pulse.register
def _(spec: CorrectedConstantPulseSpec, dac: DacChannel, name: str | None = None, base_freq: float = 0.0) -> DacPulse:
    pulse = CorrectedConstantPulse(dac, name or spec.name)
    _apply_carrier(pulse, spec, base_freq)
    pulse.length.set(spec.length)
    pulse.periodic.set(spec.periodic)

    # the three curves are interpolated together at the pulse frequency, so
    # they have to be the same length. Empty arrays mean no correction.
    freqs = np.asarray(spec.correctable_freqs, dtype=float)
    gains = np.asarray(spec.gain_factors, dtype=float)
    phases = np.asarray(spec.phase_offsets, dtype=float)
    if not (len(freqs) == len(gains) == len(phases)):
        raise ValueError(
            f"{spec.name}: correctable_freqs, gain_factors and phase_offsets must "
            f"have equal length, got {len(freqs)}, {len(gains)}, {len(phases)}"
        )
    pulse.correctable_freqs.set(freqs)
    pulse.gain_factors.set(gains)
    pulse.phase_offsets.set(phases)
    return pulse


@build_pulse.register
def _(spec: FlatTopPulseSpec, dac: DacChannel, name: str | None = None, base_freq: float = 0.0) -> DacPulse:
    pulse_name = name or spec.name
    envelope = build_envelope(spec.envelope, dac, f"{pulse_name}_envelope")
    pulse = FlatTopPulse(dac, pulse_name, envelope)
    _apply_carrier(pulse, spec, base_freq)
    # the FLAT part only; the ramps add the envelope's length on top
    pulse.length.set(spec.length)
    return pulse


@build_pulse.register
def _(spec: MuxedConstantPulseSpec, dac: MultiplexedDacChannel, name: str | None = None, base_freq: float = 0.0) -> DacPulse:
    if not hasattr(dac, "tones"):
        raise TypeError(
            f"{spec.name}: a muxed pulse needs a MultiplexedDacChannel, but DAC "
            f"{dac.channel_num} is a {type(dac).__name__}. The firmware must be "
            f"built with a muxed generator."
        )
    n_tones = len(dac.tones)
    bad = [tone for tone in spec.tone_nums if not 0 <= tone < n_tones]
    if bad:
        raise ValueError(
            f"{spec.name}: tone numbers {bad} out of range for DAC "
            f"{dac.channel_num}, which has {n_tones} tones"
        )
    # no carrier: the frequencies and gains live on the DAC's tones
    pulse = MuxedConstantPulse(dac, name or spec.name)
    pulse.length.set(spec.length)
    pulse.tone_nums.set(tuple(spec.tone_nums))
    return pulse
