"""Live elements: the qickodes objects that a spec turns into.

An element owns one DAC, optionally one ADC, and the pulses that play on it.
Unlike the specs, these need a connected QickInstrument to exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qickodes.channels_v2 import AdcChannel, DacChannel
from qickodes.pulse_base_v2 import DacPulse

from .specs import ElementSpec


@dataclass
class Element:
    """A configured element and its pulses.

    Attributes
    ----------
    name : str
        Name of the element, as written in the config.
    type : str
        "qubit" or "resonator".
    dac : DacChannel
        The DAC that plays this element's pulses.
    frequency : float
        Carrier frequency in Hz. Pulses sit at frequency + their detuning.
    adc : AdcChannel or None
        The ADC that listens to this element. None for a qubit.
    time_of_flight : float
        Round-trip delay in seconds from the DAC output back to the ADC input.
        The readout sequence opens the acquisition window this long after the
        pulse is played. Zero for a qubit.
    pulses : dict[str, DacPulse]
        Pulses of this element, by name.
    readout : Element or None
        For a qubit, the resonator element that reads it out, resolved from
        the spec's `readout` name. None for a resonator, or a qubit without one.
    spec : ElementSpec or None
        The spec this element was built from. It is the SAME object that lives
        in `Config.elements`, so editing it also updates what gets written into
        the dataset snapshot. This is what makes a calibration persist.
    """

    name: str
    type: str
    dac: DacChannel
    frequency: float
    adc: AdcChannel | None = None
    time_of_flight: float = 0.0 # useful for resonators, zero for qubits (no ADC)
    pulses: dict[str, DacPulse] = field(default_factory=dict)
    spec: ElementSpec | None = None
    readout: Element | None = None     # the resonator element that reads this one out

    def resolve_readout(self, resonator: Element | None = None) -> Element:
        """The resonator to read this element out with.

        An explicit `resonator` wins; otherwise the one named in the qubit's
        spec. This is what lets a program take a qubit alone.
        """
        if resonator is not None:
            return resonator
        if self.readout is None:
            raise ValueError(
                f"element '{self.name}' has no readout resonator. Set readout= "
                f"in its QubitSpec, or pass the resonator explicitly."
            )
        return self.readout

    def pulse(self, name: str) -> DacPulse:
        """Return one pulse by name, with a clear error if it is not there."""
        if name not in self.pulses:
            raise KeyError(
                f"element '{self.name}' has no pulse '{name}'. "
                f"It has: {sorted(self.pulses)}"
            )
        return self.pulses[name]

    @property
    def qi(self):
        return self.dac.parent # reference to the QickInstrument that owns this element

    # ------------------------------------------------------------------
    # Readout buffer.
    #
    # NOTE ON UNITS: QICK works in microseconds and megahertz, this package
    # works in seconds and hertz. Every call into raw QICK needs the * 1e6.
    # ------------------------------------------------------------------

    @property
    def buf_maxlen(self) -> int:
        """Number of samples the decimated buffer of this element's ADC holds.

        Hard limit on a decimated acquisition: the buffer has to fit `n_shots`
        windows, so a longer window leaves room for fewer repetitions. The
        accumulated path uses a different buffer ('avg_maxlen').
        """
        if self.adc is None:
            raise ValueError(
                f"element '{self.name}' has no ADC, so it has no readout buffer."
            )
        return self.qi.soccfg["readouts"][self.adc.channel_num]["buf_maxlen"]

    def window_samples(self, window_length: float) -> int:
        """Decimated samples in a window of `window_length` seconds.

        The decimated buffer holds `buf_maxlen` samples in total, so
        `buf_maxlen // window_samples(w)` is the most repetitions that fit.
        """
        if self.adc is None:
            raise ValueError(f"element '{self.name}' has no ADC.")
        samples = self.qi.soccfg.us2cycles(
            window_length * 1e6, ro_ch=self.adc.channel_num
        )
        if samples < 1:
            raise ValueError(
                f"window {window_length:.2e} s rounds to zero samples on ADC "
                f"{self.adc.channel_num}."
            )
        return samples
