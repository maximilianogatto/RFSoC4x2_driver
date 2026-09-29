import numpy as np
import matplotlib.pyplot as plt

from qcodes.dataset import (load_by_id)

from ..rfsoc import RFSoC, RunConfig
from ..elements import Element
from ..sequences import readout
from ..sweeps import restored

# software sweeps
from qickodes.instrument_v2 import SoftwareSweep

# hardware sweeps
# from qick.asm_v2 import QickSweep1D
# The readout frequency cannot be a hardware sweep: AdcChannel.freq is a
# ManualParameter, because the down-conversion frequency is a configuration
# and not a tProc register. It has to move together with the pulse frequency,
# which is what the two-parameter SoftwareSweep below does.


def resonator_spectroscopy(rfsoc: RFSoC, resonator: Element, f_start: float, f_stop: float, f_points: int, pulse_name: str = 'readout', gain: float | None = None, hard_avg: int = 1000, soft_avg: int = 1, final_delay: float = 5e-6, plot: bool = True, verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Perform a resonator spectroscopy measurement.

    Sweeps the readout tone and the ADC together across the range and records
    one IQ point per frequency. The resonance shows up as a dip (hanger or
    notch geometry) or a peak (reflection) in |IQ|, with a phase roll.

    Everything except the sweep comes from the config: the DAC, the ADC, the
    Nyquist zone, the pulse length and the readout window were all applied when
    the RFSoC was built.

    Args:
        rfsoc: RFSoC instance
        resonator: Element instance, must have an ADC
        f_start: start frequency (Hz)
        f_stop: stop frequency (Hz)
        f_points: number of frequency points
        pulse_name: name of the readout pulse
        gain: readout gain for this scan only (-1..1). None keeps the configured one.
        hard_avg: hardware averages (qi.hard_avgs), averaged on the FPGA
        soft_avg: software averages, the whole scan repeated and averaged
        final_delay: delay between shots (s). Only the resonator rings down
            here, so a few microseconds is plenty.
        plot: whether to plot the results
        verbose: whether to print the scan settings and the candidate resonance

    Returns:
        frequencies (Hz), complex IQ data
    """
    if resonator.adc is None:
        raise ValueError(f"element '{resonator.name}' has no ADC, it cannot be read out")
    if f_points < 2:
        raise ValueError(f"f_points must be at least 2, got {f_points}")
    if f_stop <= f_start:
        raise ValueError(f"f_stop ({f_stop:.4e}) must be above f_start ({f_start:.4e})")


    # configure the readout pulse and ADC
    readout_pulse = resonator.pulses[pulse_name]
    readout_adc = resonator.adc

    step = (f_stop - f_start) / (f_points - 1)
    shot = readout_pulse.length.get() + final_delay
    # a software sweep re-uploads the program at every point, ~100 ms each;
    # soft_avgs repeats the run without re-uploading
    seconds = f_points * hard_avg * soft_avg * shot + f_points * 0.1
    rfsoc.display(f"RES SPEC: {resonator.dac.channel_num}:{resonator.adc.channel_num} \ntime: {seconds/60:.1f} min")

    if verbose:
        print(f"resonator        : {resonator.name} (DAC {resonator.dac.channel_num}, ADC {readout_adc.channel_num})")
        print(f"span             : {f_start/1e9:.6f} - {f_stop/1e9:.6f} GHz, {f_points} points ({step/1e3:.1f} kHz step)")
        print(f"readout pulse    : {readout_pulse.length.get()*1e9:.0f} ns, gain {gain if gain is not None else readout_pulse.gain.get()}")
        print(f"readout window   : {readout_adc.length.get()*1e9:.0f} ns -> ~{1/readout_adc.length.get()/1e3:.0f} kHz resolution")
        print(f"averaging        : {hard_avg} hardware x {soft_avg} software")
        print(f"uploads          : {f_points}")
        print(f"estimated time   : {seconds/60:.1f} min")

    # The sweep parks the pulse and the ADC at f_stop, and `gain` is a scan
    # setting rather than a config change, so put all three back afterwards.
    with restored(readout_pulse.freq, readout_adc.freq, readout_pulse.gain):
        if gain is not None:
            readout_pulse.gain.set(gain)

        run_config = RunConfig(
            measurement_name=f"spectroscopy_{resonator.name}_{f_start:.4e}_{f_stop:.4e}",
            # one experiment per sample, so a cooldown's runs stay together
            experiment_name=f"{resonator.name}",
            sample_name=f"{resonator.name}",
            acquisition_mode='accumulated',
            n_shots=hard_avg,
            soft_avgs=soft_avg,
            final_delay=final_delay,
            # both dials move together: the ADC down-converts at the frequency
            # the pulse is sent at. Only parameters[0] is stored as the
            # setpoint, so the pulse goes first.
            software_sweeps=[SoftwareSweep([readout_pulse.freq, readout_adc.freq],
                                           f_start, f_stop, f_points)],
        )

        run_id = rfsoc.run(readout(resonator), run_config)

    # ---------------- recover ----------------
    dataset = load_by_id(run_id)
    data = dataset.get_parameter_data()

    name = list(data)[0]                  # 'iq', or 'iq_ch2' with several ADCs
    block = data[name]
    iq = np.asarray(block[name]).ravel()
    # the setpoint is the FIRST parameter of the sweep, i.e. the pulse frequency
    freq_key = next(key for key in block if key != name)
    freqs = np.asarray(block[freq_key]).ravel()

    order = np.argsort(freqs)             # a software sweep is already ordered,
    freqs, iq = freqs[order], iq[order]   # but do not rely on it

    amplitude = np.abs(iq)
    # works whether the resonance is a dip (hanger, notch) or a peak (reflection)

    # The run's snapshot holds the settings; this holds what the run CONCLUDED,
    # so the answer can be looked up later without re-analysing the trace.
    dataset.add_metadata("f_start", float(f_start))
    dataset.add_metadata("f_stop", float(f_stop))
    dataset.add_metadata("f_points", int(f_points))

    if verbose:
        contrast = amplitude.max() / amplitude.min() if amplitude.min() > 0 else np.inf
        print(f"stored as run    : {run_id}")

    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax[0].plot(freqs / 1e9, amplitude, '.-', lw=1, ms=3)
        ax[0].set_ylabel("|IQ| [ADC units]")
        ax[0].legend()
        ax[1].plot(freqs / 1e9, np.rad2deg(np.unwrap(np.angle(iq))), '.-', lw=1, ms=3)
        ax[1].set_ylabel("Phase [degrees]")
        ax[1].set_xlabel("Frequency [GHz]")
        fig.suptitle(f"{dataset.name} (run {run_id})")
        plt.tight_layout()

    rfsoc.display_ready()

    return freqs, iq
