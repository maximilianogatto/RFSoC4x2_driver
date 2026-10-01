import numpy as np
import matplotlib.pyplot as plt

from qcodes.dataset import load_by_id

from ..rfsoc import RFSoC, RunConfig
from ..elements import Element
from ..sweeps import restored

# hardware sweeps
from qick.asm_v2 import QickSweep1D
from qickodes.macros_v2 import ConfigReadout, DelayAuto, PlayPulse, Trigger
from qickodes.readout_window_v2 import ReadoutWindow

# How long before the trigger the readout frequency is sent to the ADC (s). The
# config is sent at t = 0 and the trigger fires at `time_of_flight`, so the
# time of flight has to be at least this long.
MIN_CONFIG_TO_TRIGGER = 50e-9


def remove_delay(freqs: np.ndarray, iq: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Remove the linear phase ramp (electrical delay) from IQ data sorted by frequency.

    The ramp is the median phase advance between neighbouring points, so it is
    not thrown off by a resonance that winds the phase by 360 degrees, which a
    straight-line fit of the unwrapped phase is.

    Needs a frequency step smaller than 1 / (2 * delay), otherwise the phase
    between neighbouring points is ambiguous.

    Returns:
        phase in degrees with the ramp removed, slope (deg/Hz), offset (deg)
    """
    step = np.angle(iq[1:] / iq[:-1])
    slope = np.median(step / np.diff(freqs))               # rad/Hz
    corrected = np.unwrap(np.angle(iq * np.exp(-1j * slope * (freqs - freqs[0]))))
    offset = corrected[0]
    return np.rad2deg(corrected - offset), float(np.rad2deg(slope)), float(np.rad2deg(offset))


def resonator_spectroscopy_hw(rfsoc: RFSoC, resonator: Element, f_start: float, f_stop: float, f_points: int, pulse_name: str = 'readout', gain: float | None = None, hard_avg: int = 1000, soft_avg: int = 1, final_delay: float = 5e-6, plot: bool = True, verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Resonator spectroscopy with the whole frequency sweep in ONE program.

    Same inputs and outputs as `resonator_spectroscopy`, but the readout tone
    and the ADC down-conversion frequency are swept together by the tProc, in
    hardware, instead of re-uploading the program at every frequency (~0.1 s
    each). The ADC frequency can be swept because it is sent by the tProc as a
    readout config (qickodes `ReadoutWindow` + `ConfigReadout`).

    The time per point is then just the integration: hard_avg x (pulse +
    final_delay), so it pays off most for short windows and few averages.

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
        final_delay: delay between shots (s)
        plot: whether to plot the results
        verbose: whether to print the scan settings

    Returns:
        frequencies (Hz), complex IQ data
    """
    if resonator.adc is None:
        raise ValueError(f"element '{resonator.name}' has no ADC, it cannot be read out")
    if f_points < 2:
        raise ValueError(f"f_points must be at least 2, got {f_points}")
    if f_stop <= f_start:
        raise ValueError(f"f_stop ({f_stop:.4e}) must be above f_start ({f_start:.4e})")

    readout_pulse = resonator.pulses[pulse_name]
    readout_adc = resonator.adc
    qi = resonator.qi

    # one accumulated value per frequency has to fit in the ADC's averaging buffer
    avg_maxlen = qi.soccfg["readouts"][readout_adc.channel_num]["avg_maxlen"]
    if f_points > avg_maxlen:
        raise ValueError(f"f_points = {f_points} does not fit in the accumulation buffer "
                         f"({avg_maxlen} points). Split the scan.")
    if resonator.time_of_flight < MIN_CONFIG_TO_TRIGGER:
        raise ValueError(
            f"time_of_flight is {resonator.time_of_flight*1e9:.0f} ns: the readout frequency is sent "
            f"{MIN_CONFIG_TO_TRIGGER*1e9:.0f} ns before the trigger, so it must be at least that. "
            f"Run tof_calibration first."
        )

    step = (f_stop - f_start) / (f_points - 1)
    shot = readout_pulse.length.get() + final_delay
    seconds = f_points * hard_avg * soft_avg * shot + 0.2 * soft_avg
    rfsoc.display(f"RES SPEC HW: {resonator.dac.channel_num}:{resonator.adc.channel_num} \ntime: {seconds/60:.1f} min")

    if verbose:
        print(f"resonator        : {resonator.name} (DAC {resonator.dac.channel_num}, ADC {readout_adc.channel_num})")
        print(f"span             : {f_start/1e9:.6f} - {f_stop/1e9:.6f} GHz, {f_points} points ({step/1e3:.1f} kHz step)")
        print(f"readout pulse    : {readout_pulse.length.get()*1e9:.0f} ns, gain {gain if gain is not None else readout_pulse.gain.get()}")
        print(f"readout window   : {readout_adc.length.get()*1e9:.0f} ns -> ~{1/readout_adc.length.get()/1e3:.0f} kHz bandwidth")
        print(f"averaging        : {hard_avg} hardware x {soft_avg} software")
        print(f"sweep            : in hardware, 1 upload (instead of {f_points})")
        print(f"estimated time   : {seconds:.1f} s")

    # The readout frequency lives on a ReadoutWindow, which the ConfigReadout
    # macro sends to the ADC inside the loop. The pulse and the window use the
    # SAME sweep name and range, so the tProc moves them together.
    window = ReadoutWindow(readout_adc, f"{readout_adc.short_name}_sweep")

    with restored(readout_pulse.freq, readout_adc.freq, readout_pulse.gain):
        try:
            if gain is not None:
                readout_pulse.gain.set(gain)
            readout_pulse.freq.set(QickSweep1D("freq", f_start, f_stop))
            window.freq.set(QickSweep1D("freq", f_start, f_stop))

            macros = [
                DelayAuto(qi, t=10e-9),
                ConfigReadout(qi, window, t=0),
                Trigger(qi, readout_adc, t=resonator.time_of_flight),
                PlayPulse(qi, readout_pulse),
                DelayAuto(qi, t=10e-9),
            ]

            run_config = RunConfig(
                measurement_name=f"spectroscopy_hw_{resonator.name}_{f_start:.4e}_{f_stop:.4e}",
                experiment_name=f"{resonator.name}",
                sample_name=f"{resonator.name}",
                acquisition_mode='accumulated',
                n_shots=hard_avg,
                soft_avgs=soft_avg,
                final_delay=final_delay,
                hardware_loop_counts={"freq": f_points},
            )
            run_id = rfsoc.run(macros, run_config)
        finally:
            # a swept parameter must go back to a plain number, or qickodes keeps
            # treating it as a sweep in the next program
            window.freq.set(float(f_start))

    # ---------------- recover ----------------
    dataset = load_by_id(run_id)
    data = dataset.get_parameter_data()

    name = list(data)[0]                  # 'iq', or 'iq_ch2' with several ADCs
    block = data[name]
    iq = np.asarray(block[name]).ravel()
    # two setpoints were stored (pulse and ADC frequency); they are the same grid
    freq_keys = [key for key in block if key != name]
    freq_key = next((key for key in freq_keys if f"dac{resonator.dac.channel_num}" in key), freq_keys[0])
    freqs = np.asarray(block[freq_key]).ravel()

    order = np.argsort(freqs)
    freqs, iq = freqs[order], iq[order]

    amplitude = np.abs(iq)
    phase_corrected, slope, offset = remove_delay(freqs, iq)

    if verbose:
        print(f"phase correction: {slope:.4e} deg/Hz, {offset:.4f} deg offset")
        print(f"stored as run    : {run_id}")

    dataset.add_metadata("f_start", float(f_start))
    dataset.add_metadata("f_stop", float(f_stop))
    dataset.add_metadata("f_points", int(f_points))
    dataset.add_metadata("phase_correction_slope", slope)
    dataset.add_metadata("phase_correction_offset", offset)
    dataset.add_metadata("sweep", "hardware")

    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax[0].plot(freqs / 1e9, amplitude, '.-', lw=1, ms=3)
        ax[0].set_ylabel("|IQ| [ADC units]")
        ax[1].plot(freqs / 1e9, phase_corrected, '.-', lw=1, ms=3)
        ax[1].set_ylabel("Phase [degrees]")
        ax[1].set_xlabel("Frequency [GHz]")
        fig.suptitle(f"{dataset.name} (run {run_id})")
        plt.tight_layout()

    rfsoc.display_ready()

    return freqs, iq
