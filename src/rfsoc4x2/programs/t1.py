import numpy as np
import matplotlib.pyplot as plt

from qcodes.dataset import (load_by_id)

from ..rfsoc import RFSoC, RunConfig
from ..elements import Element
from ..sequences import play, readout, wait
from ..sweeps import restored
from .rabi import project_iq
from .single_shot import to_probability

# hardware sweeps
from qick.asm_v2 import QickSweep1D


def fit_exponential(t: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit y = A exp(-t / T1) + B. Returns (T1, A, B).

    For a fixed T1 the model is LINEAR in A and B, so those come from a plain
    least-squares solve. Only T1 has to be searched: a coarse log-spaced grid,
    then a finer one around the best point. No scipy, and no starting guess to
    get wrong.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    def residual(tau):
        basis = np.column_stack([np.exp(-t / tau), np.ones_like(t)])
        coef, *_ = np.linalg.lstsq(basis, y, rcond=None)
        return np.sum((basis @ coef - y) ** 2), coef

    span = t.max() - t.min()
    step = np.min(np.diff(np.unique(t)))
    grid = np.logspace(np.log10(step / 2), np.log10(span * 20), 400)
    for _ in range(3):   # zoom in around the best point
        errors = [residual(tau)[0] for tau in grid]
        best = int(np.argmin(errors))
        lo, hi = grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)]
        grid = np.linspace(lo, hi, 200)
    t1 = float(grid[int(np.argmin([residual(tau)[0] for tau in grid]))])
    _, (amplitude, offset) = residual(t1)
    return t1, float(amplitude), float(offset)


def t1(rfsoc: RFSoC, qubit: Element, resonator: Element | None = None, delay_start: float = 0.0, delay_stop: float = 200e-6, points: int = 101, pulse_name: str = 'pi', calibration: dict | None = None, hard_avg: int = 1000, soft_avg: int = 1, final_delay: float = 500e-6, plot: bool = True, verbose: bool = True) -> tuple[np.ndarray, np.ndarray, float]:
    """Energy relaxation time T1: pi pulse, wait, read out.

    The pi pulse puts the qubit in |1>. During the wait it decays to |0>, so
    the excited population falls as exp(-t / T1). The wait is the `t` of a
    DelayAuto, which is a SweepableParameter, so the whole scan is one hardware
    loop and one upload.

    Two choices decide whether the number is right:

      delay_stop   should reach 3 to 5 x T1, so the tail and the offset it
                   decays to are both measured. Too short and the fit cannot
                   tell a slow decay from an offset.
      final_delay  the time between shots, at least 5 x T1. Too short and the
                   qubit starts a shot still partly excited, which lowers the
                   contrast AND biases the fitted T1 low, with no warning.
                   After the fit, both are checked against the measured T1.

    Args:
        rfsoc: RFSoC instance
        qubit: the element to drive
        resonator: the element to read out. Defaults to QubitSpec.readout.
        delay_start, delay_stop: wait between the pi pulse and the readout (s)
        points: points in the hardware loop
        pulse_name: the calibrated pi pulse
        calibration: the dict returned by single_shot_readout. With it the
            result is an excited-state POPULATION; without it, arbitrary units
            along the readout axis, which give the same T1.
        hard_avg: hardware averages, averaged on the FPGA
        soft_avg: software averages
        final_delay: delay between shots (s), at least 5 x T1
        plot: whether to plot
        verbose: whether to print the settings and the result

    Returns:
        delays (s), the signal (population if `calibration` was given), and T1 (s)
    """
    resonator = qubit.resolve_readout(resonator)
    if resonator.adc is None:
        raise ValueError(f"element '{resonator.name}' has no ADC, it cannot be read out")
    if points < 5:
        raise ValueError(f"points must be at least 5 to fit a decay, got {points}")
    if delay_stop <= delay_start or delay_start < 0:
        raise ValueError(f"need 0 <= delay_start < delay_stop, got {delay_start}, {delay_stop}")

    drive = qubit.pulses[pulse_name]
    if hasattr(drive, "length"):
        pi_length = drive.length.get()
    elif hasattr(drive, "envelope"):
        pi_length = drive.envelope.length.get()   # an arbitrary pulse lasts as long as its envelope
    else:
        pi_length = 0.0

    shot = pi_length + (delay_start + delay_stop) / 2 + resonator.adc.length.get() + final_delay
    seconds = points * hard_avg * soft_avg * shot + 0.1
    if verbose:
        print(f"qubit            : {qubit.name}, pulse '{pulse_name}' ({pi_length*1e9:.0f} ns), "
              f"read out by {resonator.name}")
        print(f"delay            : {delay_start*1e6:.1f} - {delay_stop*1e6:.1f} us, {points} points, hardware loop")
        print(f"final_delay      : {final_delay*1e6:.0f} us")
        print(f"averaging        : {hard_avg} hardware x {soft_avg} software")
        print(f"estimated time   : {seconds/60:.1f} min")

    rfsoc.display(f"T1: {qubit.name}\ntime: {seconds/60:.1f} min")

    # The delay macro is built with a plain number and swept inside restored():
    # a SweepableParameter stays registered as swept on the instrument until a
    # number is put back, and a stale 'delay' sweep would be picked up again by
    # the next run that uses a loop with the same name.
    delay = wait(qubit, 0.0)[0]
    with restored(delay.t):
        delay.t.set(QickSweep1D("delay", delay_start, delay_stop))

        run_config = RunConfig(
            measurement_name=f"t1_{qubit.name}",
            experiment_name=f"{qubit.name}",
            sample_name=f"{qubit.name}",
            acquisition_mode='accumulated',
            n_shots=hard_avg,
            soft_avgs=soft_avg,
            final_delay=final_delay,
            hardware_loop_counts={"delay": points},
        )

        # play() and readout() each add a 10 ns DelayAuto, so the real wait is
        # the swept one plus a constant ~20 ns. A constant offset in time only
        # rescales the amplitude of an exponential; T1 is unaffected.
        macros = play(qubit, pulse_name) + [delay] + readout(resonator)
        run_id = rfsoc.run(macros, run_config)

    # ---------------- recover ----------------
    dataset = load_by_id(run_id)
    data = dataset.get_parameter_data()

    name = list(data)[0]
    block = data[name]
    iq = np.asarray(block[name]).ravel()
    delay_key = next(key for key in block if key != name)
    delays = np.asarray(block[delay_key]).ravel()

    order = np.argsort(delays)
    delays, iq = delays[order], iq[order]

    if calibration is not None:
        signal = to_probability(iq, calibration["ground"], calibration["excited"])
        ylabel = "excited-state population"
    else:
        signal = project_iq(iq)
        ylabel = "projected signal [ADC units]"

    t1_value, amplitude, offset = fit_exponential(delays, signal)

    dataset.add_metadata("t1", t1_value)
    dataset.add_metadata("t1_amplitude", amplitude)
    dataset.add_metadata("t1_offset", offset)

    if verbose:
        print(f"stored as run    : {run_id}")
        print(f"T1               : {t1_value*1e6:.2f} us")
        if delay_stop < 3 * t1_value:
            print(f"WARNING: the scan only reaches {delay_stop/t1_value:.1f} x T1. Extend delay_stop "
                  f"to ~{5*t1_value*1e6:.0f} us so the fit sees the tail.")
        if final_delay < 5 * t1_value:
            print(f"WARNING: final_delay is {final_delay/t1_value:.1f} x T1. Below 5 x T1 the qubit "
                  f"starts some shots still excited, which biases T1 low.")
            print(f"         Rerun with final_delay >= {5*t1_value*1e6:.0f} us.")
        if (delays[1] - delays[0]) > t1_value / 3:
            print("WARNING: the step is coarse compared with T1; use more points or a shorter range.")

    rfsoc.display_ready()

    if plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(delays * 1e6, signal, '.', ms=4, label="data")
        fine = np.linspace(delays.min(), delays.max(), 400)
        ax.plot(fine * 1e6, amplitude * np.exp(-fine / t1_value) + offset, 'r-',
                label=f"T1 = {t1_value*1e6:.2f} us")
        ax.set_xlabel("Delay [us]")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.suptitle(f"{dataset.name} (run {run_id})")
        plt.tight_layout()

    return delays, signal, t1_value
