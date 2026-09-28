import numpy as np
import matplotlib.pyplot as plt

from qcodes.dataset import (load_by_id)

from ..rfsoc import RFSoC, RunConfig
from ..elements import Element
from ..sequences import play, readout
from ..sweeps import restored

# hardware sweeps
from qick.asm_v2 import QickSweep1D

# What each flavour sweeps. Both are SweepableParameters, so both run as a
# hardware loop: one upload, the tProc steps the register.
SWEEPS = {
    "gain": ("gain", "", 1.0),          # dimensionless, -1 .. 1
    "length": ("length", "ns", 1e9),    # seconds, shown in ns
}


def project_iq(iq: np.ndarray) -> np.ndarray:
    """Project complex IQ onto the axis it actually varies along.

    A qubit measurement moves the IQ point along a line between the ground and
    excited state blobs; the angle of that line is arbitrary and set by the
    cable lengths. Rotating onto it turns two noisy quadratures into one clean
    signal, without needing a fit.

    ONE angle is used for the whole array, including a 2D one: the readout
    angle is a property of the wiring, not of the sweep, so projecting each row
    separately would flip signs between rows.
    """
    array = np.asarray(iq)
    centred = array - np.mean(array)
    angle = 0.5 * np.angle(np.mean(centred.ravel() ** 2))   # largest-variance direction
    return np.real(centred * np.exp(-1j * angle))


def find_pi(x: np.ndarray, signal: np.ndarray, smooth: int = 3) -> float:
    """First turning point of a Rabi oscillation: the pi pulse.

    Half a period of the oscillation inverts the qubit, so the first extremum
    is the pi pulse. Falls back to the largest excursion if the scan does not
    reach a turning point, which means the range was too short.
    """
    if smooth > 1:
        signal = np.convolve(signal, np.ones(smooth) / smooth, mode="same")
    slope = np.diff(signal)
    turns = np.where(np.diff(np.sign(slope)) != 0)[0]
    # ignore turning points in the smoothing edge, they are artefacts
    turns = turns[turns > smooth]
    if len(turns) == 0:
        return float(x[np.argmax(np.abs(signal))])
    return float(x[turns[0] + 1])


def rabi(rfsoc: RFSoC, qubit: Element, resonator: Element, sweep: str = 'gain', start: float = 0.0, stop: float = 1.0, points: int = 101, pulse_name: str = 'pi', hard_avg: int = 1000, soft_avg: int = 1, final_delay: float = 200e-6, plot: bool = True, verbose: bool = True) -> tuple[np.ndarray, np.ndarray, float]:
    """Rabi oscillation: drive the qubit, read out, and find the pi pulse.

    Driving on resonance rotates the qubit around the Bloch sphere by an angle
    proportional to amplitude x duration, so the readout oscillates as either
    knob is turned. The first turning point is a full inversion: the pi pulse.

    There is NO wait between the drive and the readout. That is what separates
    this from T1, which inserts a delay after an already calibrated pi pulse.

    Two flavours, same sequence:

        sweep='gain'    amplitude Rabi. Works with any pulse, including a
                        shaped ArbitraryPulse, and is the usual choice once
                        the pulse shape is fixed.
        sweep='length'  time Rabi. Needs a ConstantPulse: an ArbitraryPulse
                        takes its duration from its envelope, which is not a
                        SweepableParameter.

    Args:
        rfsoc: RFSoC instance
        qubit: the element to drive
        resonator: the element to read out, must have an ADC
        sweep: 'gain' or 'length'
        start, stop: limits of the swept parameter (-1..1 for gain, seconds for length)
        points: number of points in the hardware loop
        pulse_name: which of the qubit's pulses to drive with
        hard_avg: hardware averages, averaged on the FPGA
        soft_avg: software averages
        final_delay: delay between shots (s). Must be at least 5x T1, ideally
            10x, or the qubit has not relaxed when the next shot starts.
        plot: whether to plot
        verbose: whether to print the settings and the pi pulse

    Returns:
        x (gain or seconds), complex IQ, and the pi pulse value
    """
    if sweep not in SWEEPS:
        raise ValueError(f"sweep must be one of {list(SWEEPS)}, got '{sweep}'")
    if resonator.adc is None:
        raise ValueError(f"element '{resonator.name}' has no ADC, it cannot be read out")
    if points < 2:
        raise ValueError(f"points must be at least 2, got {points}")

    drive = qubit.pulses[pulse_name]
    attribute, unit, scale = SWEEPS[sweep]
    if not hasattr(drive, attribute):
        raise TypeError(
            f"pulse '{pulse_name}' has no '{attribute}', so it cannot do a {sweep} "
            f"Rabi. A shaped pulse takes its duration from its envelope; use a "
            f"ConstantPulse for a time Rabi, or sweep='gain' instead."
        )

    # a hardware loop uploads once, however many points
    shot = (stop + start) / 2 if sweep == 'length' else drive.length.get() if hasattr(drive, 'length') else 0
    shot += resonator.adc.length.get() + final_delay
    seconds = points * hard_avg * soft_avg * shot + 0.1
    if verbose:
        print(f"qubit            : {qubit.name} at {qubit.frequency/1e9:.6f} GHz, pulse '{pulse_name}'")
        print(f"sweep            : {sweep} {start*scale:.4g} - {stop*scale:.4g} {unit}, {points} points")
        print(f"                   hardware loop, 1 upload")
        print(f"averaging        : {hard_avg} hardware x {soft_avg} software")
        print(f"final_delay      : {final_delay*1e6:.0f} us (must be >= 5x T1)")
        print(f"estimated time   : {seconds/60:.1f} min")

    parameter = getattr(drive, attribute)
    # assigning a QickSweep1D mutates the parameter, and it stays swept until a
    # number is put back. restored() snapshots first, so that is what returns.
    with restored(parameter):
        parameter.set(QickSweep1D(sweep, start, stop))

        run_config = RunConfig(
            measurement_name=f"rabi_{sweep}_{qubit.name}",
            experiment_name=f"{qubit.name}",
            sample_name=f"{qubit.name}",
            acquisition_mode='accumulated',
            n_shots=hard_avg,
            soft_avgs=soft_avg,
            final_delay=final_delay,
            hardware_loop_counts={sweep: points},
        )

        macros = play(qubit, pulse_name) + readout(resonator)
        run_id = rfsoc.run(macros, run_config)

    # ---------------- recover ----------------
    dataset = load_by_id(run_id)
    data = dataset.get_parameter_data()

    name = list(data)[0]
    block = data[name]
    iq = np.asarray(block[name]).ravel()
    x_key = next(key for key in block if key != name)
    x = np.asarray(block[x_key]).ravel()

    order = np.argsort(x)
    x, iq = x[order], iq[order]

    signal = project_iq(iq)
    pi_value = find_pi(x, signal)

    dataset.add_metadata(f"pi_{attribute}", pi_value)
    dataset.add_metadata("rabi_sweep", sweep)

    if verbose:
        print(f"stored as run    : {run_id}")
        print(f"pi pulse         : {attribute} = {pi_value*scale:.5g} {unit}")
        if pi_value in (x[0], x[-1]):
            print("WARNING: the turning point is at the edge of the scan. Widen the range:")
            print("         without a full half period this is not a pi pulse.")

    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        ax[0].plot(x * scale, signal, '.-', lw=1, ms=3)
        ax[0].axvline(pi_value * scale, color='r', ls='--',
                      label=f"pi: {pi_value*scale:.4g} {unit}")
        ax[0].set_ylabel("projected signal [ADC units]")
        ax[0].legend()
        ax[1].plot(x * scale, np.abs(iq), '.-', lw=1, ms=3, label="|IQ|")
        ax[1].set_ylabel("|IQ| [ADC units]")
        ax[1].set_xlabel(f"{attribute} [{unit}]" if unit else attribute)
        ax[1].legend()
        fig.suptitle(f"{dataset.name} (run {run_id})")
        plt.tight_layout()

    return x, iq, pi_value
