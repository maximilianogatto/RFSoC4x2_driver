import numpy as np
import matplotlib.pyplot as plt

from qcodes.dataset import (load_by_id)

from ..rfsoc import RFSoC, RunConfig
from ..elements import Element
from ..sequences import readout, play
from ..sweeps import restored
from .rabi import project_iq

# hardware sweeps
from qick.asm_v2 import QickSweep1D


def qubit_spectroscopy(rfsoc: RFSoC, qubit: Element, f_start: float, f_stop: float, f_points: int, resonator: Element | None = None, pulse_name: str = 'drive', gain: float | None = None, hard_avg: int = 1000, soft_avg: int = 1, final_delay: float = 200e-6, plot: bool = True, verbose: bool = True) -> tuple[np.ndarray, np.ndarray, float]:
    """Perform a qubit spectroscopy measurement (two-tone, or "pulse probe").

    Drives the qubit with a long, weak pulse at a swept frequency, then reads
    out through its resonator at the fixed, calibrated readout frequency. When
    the drive hits the qubit transition the qubit is partly excited, the
    resonator shifts dispersively, and the readout IQ moves. Off resonance
    nothing happens, so the spectrum is flat except at the qubit frequency.

    The drive frequency is a HARDWARE sweep: no ADC has to follow it, unlike a
    readout sweep, so the whole scan is one upload.

    The drive should be a long, weak ConstantPulse (a "probe" or saturation
    pulse), not the pi pulse: the qubit frequency is not known yet, so there is
    no calibrated pi pulse. Too strong and power broadening hides the line; too
    weak and it is not visible. The qickodes example uses 100 us at gain 0.2.

    Everything except the sweep comes from the config: the DAC, the Nyquist
    zone, the pulse length, the readout pulse, the readout window and the time
    of flight were all applied when the RFSoC was built.

    Args:
        rfsoc: RFSoC instance
        qubit: the element to drive
        f_start: start of the drive frequency scan (Hz)
        f_stop: end of the drive frequency scan (Hz)
        f_points: number of frequency points
        resonator: the element to read out. Defaults to the resonator named in
            the qubit's spec (QubitSpec.readout).
        pulse_name: name of the drive pulse
        gain: drive gain for this scan only (-1..1). None keeps the configured one.
        hard_avg: hardware averages (qi.hard_avgs), averaged on the FPGA
        soft_avg: software averages, the whole scan repeated and averaged
        final_delay: delay between shots (s). The drive excites the qubit, so
            it has to relax before the next shot: at least 5x T1, ideally 10x.
            Too short and every shot starts partly excited, which washes out
            the line.
        plot: whether to plot the results
        verbose: whether to print the scan settings and the candidate frequency

    Returns:
        drive frequencies (Hz), complex IQ, and the candidate qubit frequency (Hz)
    """
    resonator = qubit.resolve_readout(resonator)
    if resonator.adc is None:
        raise ValueError(f"element '{resonator.name}' has no ADC, it cannot be read out")
    if qubit.adc is not None:
        raise ValueError(f"element '{qubit.name}' has an ADC: it is a resonator, not a qubit")
    if f_points < 2:
        raise ValueError(f"f_points must be at least 2, got {f_points}")
    if f_stop <= f_start:
        raise ValueError(f"f_stop ({f_stop:.4e}) must be above f_start ({f_start:.4e})")

    drive_pulse = qubit.pulses[pulse_name]

    # one upload, the tProc steps the frequency register
    drive_length = drive_pulse.length.get() if hasattr(drive_pulse, 'length') else 0.0
    shot = drive_length + resonator.adc.length.get() + final_delay
    seconds = f_points * hard_avg * soft_avg * shot + 0.1
    step = (f_stop - f_start) / (f_points - 1)
    if verbose:
        print(f"qubit            : {qubit.name} (DAC {qubit.dac.channel_num}), pulse '{pulse_name}'")
        print(f"readout          : {resonator.name} at {resonator.frequency/1e9:.6f} GHz")
        print(f"span             : {f_start/1e9:.6f} - {f_stop/1e9:.6f} GHz, {f_points} points ({step/1e3:.1f} kHz step)")
        print(f"drive            : {drive_length*1e6:.1f} us, gain {gain if gain is not None else drive_pulse.gain.get()}")
        print(f"sweep            : hardware loop, 1 upload")
        print(f"averaging        : {hard_avg} hardware x {soft_avg} software")
        print(f"final_delay      : {final_delay*1e6:.0f} us (must be >= 5x T1)")
        print(f"estimated time   : {seconds/60:.1f} min")

    rfsoc.display(f"QUBIT SPEC: {qubit.name}\ntime: {seconds/60:.1f} min")

    # restored() snapshots FIRST, so assigning the sweep has to happen inside
    # the block: taken outside, the snapshot would be the sweep object itself
    # and the parameter would never get its number back. `gain` is a scan
    # setting, so it goes back too.
    with restored(drive_pulse.freq, drive_pulse.gain):
        drive_pulse.freq.set(QickSweep1D('freq', f_start, f_stop))
        if gain is not None:
            drive_pulse.gain.set(gain)

        run_config = RunConfig(
            measurement_name=f"qubit_spectroscopy_{qubit.name}_{f_start:.4e}_{f_stop:.4e}",
            # one experiment per sample, so a cooldown's runs stay together
            experiment_name=f"{qubit.name}",
            sample_name=f"{qubit.name}",
            acquisition_mode='accumulated',
            n_shots=hard_avg,
            soft_avgs=soft_avg,
            final_delay=final_delay,
            # only the drive moves: the readout stays at its calibrated
            # frequency, so nothing else has to follow
            hardware_loop_counts={'freq': f_points},
        )

        macros = play(qubit, pulse_name) + readout(resonator)
        run_id = rfsoc.run(macros, run_config)

    # ---------------- recover ----------------
    dataset = load_by_id(run_id)
    data = dataset.get_parameter_data()

    name = list(data)[0]                  # 'iq', or 'iq_ch2' with several ADCs
    block = data[name]
    iq = np.asarray(block[name]).ravel()
    freq_key = next(key for key in block if key != name)
    freqs = np.asarray(block[freq_key]).ravel()

    order = np.argsort(freqs)
    freqs, iq = freqs[order], iq[order]

    # The line can move the IQ point in any direction depending on where the
    # readout sits, so neither |IQ| nor the phase alone is reliable. Project
    # onto the axis the data actually moves along, then take the biggest
    # departure from the flat background.
    signal = project_iq(iq)
    f_qubit = float(freqs[np.argmax(np.abs(signal - np.median(signal)))])

    dataset.add_metadata("qubit_frequency", f_qubit)
    dataset.add_metadata("f_start", float(f_start))
    dataset.add_metadata("f_stop", float(f_stop))

    if verbose:
        print(f"stored as run    : {run_id}")
        print(f"candidate f_q    : {f_qubit/1e9:.6f} GHz ({(f_qubit - qubit.frequency)/1e6:+.3f} MHz from the config)")
        if f_qubit in (freqs[0], freqs[-1]):
            print("WARNING: the extremum sits at the edge of the span, widen the range")
        print("If the line is real, store it and refine it with a chevron:")
        print(f"  rfsoc.update_spec([('frequency', {f_qubit:.6e})], element='{qubit.name}')")

    rfsoc.display_ready()

    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax[0].plot(freqs / 1e9, signal, '.-', lw=1, ms=3)
        ax[0].axvline(f_qubit / 1e9, color='r', ls='--',
                      label=f"f_q: {f_qubit/1e9:.6f} GHz")
        ax[0].set_ylabel("projected signal [ADC units]")
        ax[0].legend()
        ax[1].plot(freqs / 1e9, np.abs(iq), '.-', lw=1, ms=3, label="|IQ|")
        ax[1].set_ylabel("|IQ| [ADC units]")
        ax[1].set_xlabel("Drive frequency [GHz]")
        ax[1].legend()
        fig.suptitle(f"{dataset.name} (run {run_id})")
        plt.tight_layout()

    return freqs, iq, f_qubit
