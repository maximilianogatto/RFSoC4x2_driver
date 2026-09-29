import numpy as np
import matplotlib.pyplot as plt

from qcodes.dataset import (load_by_id)

from ..rfsoc import RFSoC, RunConfig
from ..elements import Element
from ..sequences import readout, play
from ..sweeps import restored
from .rabi import project_iq

# software sweeps
from qickodes.instrument_v2 import SoftwareSweep
# the parameters the tProc can step: only these can be a hardware loop
from qickodes.parameters_v2 import SweepableParameter

# hardware sweeps
from qick.asm_v2 import QickSweep1D


def _is_linear(values: np.ndarray) -> bool:
    """True if the points are evenly spaced, which a hardware loop requires."""
    steps = np.diff(values)
    return bool(np.allclose(steps, steps[0], rtol=1e-6, atol=0))


def two_tone(rfsoc: RFSoC, qubit: Element, f_start: float, f_stop: float, f_points: int, parameter, values, hardware: bool | None = None, resonator: Element | None = None, pulse_name: str = 'drive', hard_avg: int = 1000, soft_avg: int = 1, final_delay: float = 200e-6, plot: bool = True, verbose: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two-tone spectroscopy against any one parameter you choose.

    The drive frequency is always the inner axis, as a hardware loop. The
    outer axis is whatever `parameter` you pass, stepped through `values`:

        the drive gain         -> the line against drive power
        the readout gain       -> the line against measurement strength
        a current source       -> the line against flux, for a tunable qubit
        anything with .set()   -> it is a QCoDeS parameter, so it works

    Hardware or software is decided by `hardware`:

        True   hardware loop. The parameter must be a qickodes
               SweepableParameter (a pulse's freq, gain, phase, length, or a
               delay) and the values evenly spaced, since QickSweep1D is
               linear. The whole map is ONE upload.
        False  software sweep. Any QCoDeS parameter and any values, e.g.
               log spaced, or an external instrument. One upload per value.
        None   hardware when it is possible, software otherwise.

    An external instrument is always software: it is not on the board, so the
    tProc cannot step it. Add it to the Station so its settings are recorded
    in the snapshot, and check whether it needs ramping - `restored()` puts it
    back to its starting value in one step at the end.

    Args:
        rfsoc: RFSoC instance
        qubit: the element to drive
        f_start, f_stop: drive frequency limits (Hz)
        f_points: points on the frequency axis
        parameter: the QCoDeS parameter of the outer axis
        values: the values it takes, in order
        hardware: True, False, or None to choose automatically
        resonator: the element to read out. Defaults to QubitSpec.readout.
        pulse_name: name of the drive pulse, a long weak ConstantPulse
        hard_avg: hardware averages, averaged on the FPGA
        soft_avg: software averages
        final_delay: delay between shots (s), at least 5x T1
        plot: whether to plot the map
        verbose: whether to print the settings and the result

    Returns:
        drive frequencies (Hz), parameter values, complex IQ of shape
        (len(values), f_points), and the line frequency at each value (Hz)
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
    if parameter is drive_pulse.freq:
        raise ValueError("the drive frequency is already the inner axis; pick another parameter")

    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError(f"values must be a 1D list of at least 2 points, got shape {values.shape}")

    # can this axis be a hardware loop at all?
    sweepable = isinstance(parameter, SweepableParameter)
    linear = _is_linear(values)
    if hardware is None:
        hardware = sweepable and linear
    elif hardware and not sweepable:
        raise ValueError(
            f"'{parameter.full_name}' is not a SweepableParameter, so the tProc cannot "
            f"step it. Use hardware=False (an external instrument is always software)."
        )
    elif hardware and not linear:
        raise ValueError(
            f"a hardware loop needs evenly spaced values (QickSweep1D is linear). "
            f"Use hardware=False for these values, e.g. log spaced."
        )

    label = parameter.label or parameter.name
    unit = parameter.unit or ""

    total = f_points * len(values)
    drive_length = drive_pulse.length.get() if hasattr(drive_pulse, 'length') else 0.0
    shot = drive_length + resonator.adc.length.get() + final_delay
    uploads = 1 if hardware else len(values)
    seconds = total * hard_avg * soft_avg * shot + uploads * 0.1
    if verbose:
        print(f"qubit            : {qubit.name}, pulse '{pulse_name}', read out by {resonator.name}")
        print(f"frequency        : {f_start/1e9:.6f} - {f_stop/1e9:.6f} GHz, {f_points} points (hardware)")
        print(f"outer axis       : {parameter.full_name}, {values[0]:.4g} - {values[-1]:.4g} {unit}, "
              f"{len(values)} points ({'hardware' if hardware else 'software'})")
        print(f"grid             : {total} points, {uploads} upload(s)")
        print(f"averaging        : {hard_avg} hardware x {soft_avg} software")
        print(f"estimated time   : {seconds/60:.1f} min")

    rfsoc.display(f"2-TONE: {qubit.name}\nvs {parameter.name}\ntime: {seconds/60:.1f} min")

    # enter restored() FIRST, then assign the sweeps inside it: taken after,
    # the snapshot would be the sweep object and never get its number back
    with restored(drive_pulse.freq, parameter):
        drive_pulse.freq.set(QickSweep1D('freq', f_start, f_stop))

        if hardware:
            # a loop name of its own, so it can never merge with 'freq'
            loop = f"axis_{parameter.name}"
            parameter.set(QickSweep1D(loop, values[0], values[-1]))
            software_sweeps = []
            # dict order is outer -> inner: one frequency scan per value
            hardware_loop_counts = {loop: len(values), 'freq': f_points}
        else:
            software_sweeps = [SoftwareSweep(parameter, values)]
            hardware_loop_counts = {'freq': f_points}

        run_config = RunConfig(
            measurement_name=f"two_tone_{qubit.name}_vs_{parameter.name}",
            experiment_name=f"{qubit.name}",
            sample_name=f"{qubit.name}",
            acquisition_mode='accumulated',
            n_shots=hard_avg,
            soft_avgs=soft_avg,
            final_delay=final_delay,
            software_sweeps=software_sweeps,
            hardware_loop_counts=hardware_loop_counts,
        )

        macros = play(qubit, pulse_name) + readout(resonator)
        run_id = rfsoc.run(macros, run_config)

    # ---------------- recover ----------------
    dataset = load_by_id(run_id)
    data = dataset.get_parameter_data()

    name = list(data)[0]
    block = data[name]
    iq_flat = np.asarray(block[name]).ravel()
    freq_key, axis_key = drive_pulse.freq.full_name, parameter.full_name
    for key in (freq_key, axis_key):
        if key not in block:
            raise KeyError(f"expected column '{key}' in the dataset, found {list(block)}")
    freq_flat = np.asarray(block[freq_key]).ravel()
    axis_flat = np.asarray(block[axis_key]).ravel()

    # rebuild the grid from the columns, whichever way the sweeps were nested
    order = np.lexsort((freq_flat, axis_flat))
    freqs = np.unique(freq_flat)
    axis = np.unique(axis_flat)
    iq = iq_flat[order].reshape(len(axis), len(freqs))

    # one projection angle for the whole map: it is set by the wiring. Then
    # the line at each value is the biggest departure from that row's median.
    signal = project_iq(iq)
    deviation = np.abs(signal - np.median(signal, axis=1, keepdims=True))
    lines = freqs[np.argmax(deviation, axis=1)]

    dataset.add_metadata("outer_axis", parameter.full_name)

    if verbose:
        print(f"stored as run    : {run_id}")
        print(f"line             : {lines.min()/1e9:.6f} - {lines.max()/1e9:.6f} GHz across the scan")

    rfsoc.display_ready()

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True,
                               gridspec_kw={'width_ratios': [3, 1]})
        mesh = ax[0].pcolormesh(freqs / 1e9, axis, signal, shading='nearest')
        ax[0].plot(lines / 1e9, axis, 'w.', ms=3, label="line")
        ax[0].set_xlabel("Drive frequency [GHz]")
        ax[0].set_ylabel(f"{label} [{unit}]" if unit else label)
        ax[0].set_title("Projected signal")
        ax[0].legend()
        fig.colorbar(mesh, ax=ax[0])

        ax[1].plot(lines / 1e9, axis, 'k.-')
        ax[1].set_xlabel("Line [GHz]")
        ax[1].set_title(f"Line vs {parameter.name}")
        ax[1].grid(alpha=0.3)

        fig.suptitle(f"{dataset.name} (run {run_id})")
        plt.tight_layout()

    return freqs, axis, iq, lines
