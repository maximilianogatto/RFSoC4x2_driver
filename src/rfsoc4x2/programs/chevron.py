import numpy as np
import matplotlib.pyplot as plt

from qcodes.dataset import (load_by_id)

from ..rfsoc import RFSoC, RunConfig
from ..elements import Element
from ..sequences import play, readout
from ..sweeps import restored
from .rabi import project_iq

# hardware sweeps
from qick.asm_v2 import QickSweep1D


def find_qubit_frequency(freqs: np.ndarray, signal: np.ndarray) -> float:
    """The frequency with the strongest oscillation: the qubit frequency.

    Off resonance the qubit rotates about a tilted axis, so it never fully
    inverts: the contrast falls as Omega^2 / (Omega^2 + Delta^2). The row with
    the largest peak-to-peak swing is therefore the vertex of the chevron.
    """
    contrast = signal.max(axis=1) - signal.min(axis=1)
    return float(freqs[np.argmax(contrast)])


def chevron(rfsoc: RFSoC, qubit: Element, resonator: Element | None = None, t_start: float = 20e-9, t_stop: float = 2e-6, t_points: int = 41, f_span: float = 20e6, f_points: int = 41, pulse_name: str = 'drive', hard_avg: int = 1000, soft_avg: int = 1, final_delay: float = 200e-6, plot: bool = True, verbose: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Time Rabi against drive detuning: the chevron, and the qubit frequency.

    A time Rabi repeated at each detuning. On resonance the qubit inverts fully
    and slowly; off resonance it rotates about a tilted axis, so the oscillation
    gets faster and shallower:

        Omega_eff = sqrt(Omega^2 + Delta^2)

    The pattern narrows into a V whose vertex sits at the TRUE qubit frequency,
    which is why a chevron is how you pin the frequency down after spectroscopy
    has given you a rough value.

    Both axes are hardware loops - drive frequency needs no ADC to follow it,
    unlike a readout sweep - so the whole 2D map is a single program upload.

    The drive must be a ConstantPulse: only a constant pulse has a sweepable
    `length`. This is usually a separate plain pulse from the shaped pi pulse
    you calibrate later.

    Args:
        rfsoc: RFSoC instance
        qubit: the element to drive
        resonator: the element to read out. Defaults to the resonator named
            in the qubit's spec (QubitSpec.readout).
        t_start, t_stop: drive duration limits (s)
        t_points: points on the duration axis
        f_span: total width of the frequency scan (Hz), centred on the qubit's
            current frequency
        f_points: points on the frequency axis
        pulse_name: which of the qubit's pulses to drive with, must be constant
        hard_avg: hardware averages, averaged on the FPGA
        soft_avg: software averages
        final_delay: delay between shots (s). At least 5x T1.
        plot: whether to plot the 2D map
        verbose: whether to print the settings and the extracted frequency

    Returns:
        durations (s), frequencies (Hz), complex IQ of shape
        (f_points, t_points), and the extracted qubit frequency (Hz)
    """
    resonator = qubit.resolve_readout(resonator)
    if resonator.adc is None:
        raise ValueError(f"element '{resonator.name}' has no ADC, it cannot be read out")
    if t_points < 2 or f_points < 2:
        raise ValueError(f"need at least 2 points per axis, got {t_points} x {f_points}")

    drive = qubit.pulses[pulse_name]
    if not hasattr(drive, "length"):
        raise TypeError(
            f"pulse '{pulse_name}' has no 'length', so its duration cannot be swept. "
            f"A chevron needs a ConstantPulse; a shaped pulse takes its duration "
            f"from its envelope."
        )

    f_start, f_stop = qubit.frequency - f_span / 2, qubit.frequency + f_span / 2

    total = t_points * f_points
    shot = (t_start + t_stop) / 2 + resonator.adc.length.get() + final_delay
    seconds = total * hard_avg * soft_avg * shot + 0.1     # one upload, both axes
    if verbose:
        print(f"qubit            : {qubit.name} at {qubit.frequency/1e9:.6f} GHz, pulse '{pulse_name}'")
        print(f"duration         : {t_start*1e9:.0f} - {t_stop*1e9:.0f} ns, {t_points} points")
        print(f"frequency        : {f_start/1e9:.6f} - {f_stop/1e9:.6f} GHz, {f_points} points")
        print(f"                   ({f_span/1e6:.1f} MHz span, {f_span/(f_points-1)/1e3:.0f} kHz step)")
        print(f"grid             : {total} points, both hardware loops, 1 upload")
        print(f"averaging        : {hard_avg} hardware x {soft_avg} software")
        print(f"estimated time   : {seconds/60:.1f} min")

    # both assignments mutate their parameter until a number is put back
    with restored(drive.length, drive.freq):
        drive.length.set(QickSweep1D("length", t_start, t_stop))
        drive.freq.set(QickSweep1D("freq", f_start, f_stop))

        run_config = RunConfig(
            measurement_name=f"chevron_{qubit.name}",
            experiment_name=f"{qubit.name}",
            sample_name=f"{qubit.name}",
            acquisition_mode='accumulated',
            n_shots=hard_avg,
            soft_avgs=soft_avg,
            final_delay=final_delay,
            # dict order is outer -> inner: one duration scan per frequency
            hardware_loop_counts={"freq": f_points, "length": t_points},
        )

        macros = play(qubit, pulse_name) + readout(resonator)
        run_id = rfsoc.run(macros, run_config)

    # ---------------- recover ----------------
    dataset = load_by_id(run_id)
    data = dataset.get_parameter_data()

    name = list(data)[0]
    block = data[name]
    iq_flat = np.asarray(block[name]).ravel()
    freq_key, length_key = drive.freq.full_name, drive.length.full_name
    for key in (freq_key, length_key):
        if key not in block:
            raise KeyError(f"expected column '{key}' in the dataset, found {list(block)}")
    freq_flat = np.asarray(block[freq_key]).ravel()
    time_flat = np.asarray(block[length_key]).ravel()

    # rebuild the grid from the columns, not from the acquisition order
    order = np.lexsort((time_flat, freq_flat))
    durations = np.unique(time_flat)
    freqs = np.unique(freq_flat)
    iq = iq_flat[order].reshape(len(freqs), len(durations))

    # one projection angle for the whole map: it is set by the wiring, not by
    # the detuning, so per-row angles would flip signs between rows
    signal = project_iq(iq)
    f_qubit = find_qubit_frequency(freqs, signal)

    dataset.add_metadata("qubit_frequency", f_qubit)
    dataset.add_metadata("f_span", float(f_span))

    if verbose:
        shift = f_qubit - qubit.frequency
        print(f"stored as run    : {run_id}")
        print(f"qubit frequency  : {f_qubit/1e9:.6f} GHz ({shift/1e6:+.3f} MHz from the config)")
        if f_qubit in (freqs[0], freqs[-1]):
            print("WARNING: the vertex is at the edge of the span, widen f_span")

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        mesh = ax[0].pcolormesh(durations * 1e9, freqs / 1e9, signal, shading='nearest')
        ax[0].axhline(f_qubit / 1e9, color='r', ls='--',
                      label=f"{f_qubit/1e9:.6f} GHz")
        ax[0].set_xlabel("Drive duration [ns]")
        ax[0].set_ylabel("Drive frequency [GHz]")
        ax[0].set_title("Chevron")
        ax[0].legend()
        fig.colorbar(mesh, ax=ax[0])

        ax[1].plot(durations * 1e9, signal[np.argmax(signal.max(axis=1) - signal.min(axis=1))],
                   '.-', lw=1, ms=3)
        ax[1].set_xlabel("Drive duration [ns]")
        ax[1].set_ylabel("projected signal [ADC units]")
        ax[1].set_title("On resonance: a time Rabi")
        ax[1].grid(alpha=0.3)

        fig.suptitle(f"{dataset.name} (run {run_id})")
        plt.tight_layout()

    return durations, freqs, iq, f_qubit
