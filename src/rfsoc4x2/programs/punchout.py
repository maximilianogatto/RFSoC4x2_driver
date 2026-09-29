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
from qick.asm_v2 import QickSweep1D


def find_resonances(freqs: np.ndarray, iq: np.ndarray) -> np.ndarray:
    """One resonance per power, from a (gains, freqs) grid of IQ.

    Takes the biggest departure from each row's own median, which works for a
    dip (hanger, notch) or a peak (reflection). Separate from the measurement
    so an old run can be re-analysed without re-measuring.
    """
    amplitude = np.abs(np.atleast_2d(iq))
    baseline = np.median(amplitude, axis=1, keepdims=True)
    return freqs[np.argmax(np.abs(amplitude - baseline), axis=1)]


def plot_punchout(freqs, gains, iq, resonances, log_gain=True, title=""):
    """Map of |IQ| against frequency and power, plus the punch out curve.

    Each row is normalised by its own median: the raw amplitude spans decades
    because the gain does, which would hide the dip at low power.
    """
    amplitude = np.abs(iq)
    normalised = amplitude / np.median(amplitude, axis=1, keepdims=True)
    scale = 'log' if log_gain else 'linear'

    fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    mesh = ax[0].pcolormesh(freqs / 1e9, gains, normalised, shading='nearest')
    ax[0].plot(resonances / 1e9, gains, 'r.-', lw=1, ms=4, label="resonance")
    ax[0].set_yscale(scale)
    ax[0].set_xlabel("Frequency [GHz]")
    ax[0].set_ylabel("Readout gain")
    ax[0].set_title("|IQ| normalised per power")
    ax[0].legend()
    fig.colorbar(mesh, ax=ax[0])

    ax[1].plot(resonances / 1e9, gains, 'k.-')
    ax[1].set_yscale(scale)
    ax[1].set_xlabel("Frequency [GHz]")
    ax[1].set_title("Resonance vs power (the punch out)")
    ax[1].grid(alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    return fig


def punchout(rfsoc: RFSoC, resonator: Element, f_start: float, f_stop: float, f_points: int, gain_start: float = 1e-3, gain_stop: float = 1.0, gain_points: int = 21, gain_spacing: str = 'linear', pulse_name: str = 'readout', hard_avg: int = 1000, soft_avg: int = 1, final_delay: float = 5e-6, plot: bool = True, verbose: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Measure the resonator against readout power.

    A resonator coupled to a qubit sits at two different frequencies depending
    on the drive power. At LOW power it is the dressed resonance, pulled by the
    qubit by roughly g^2/delta. At HIGH power the qubit saturates and the
    resonance snaps back to the bare cavity. The transition is the "punch out",
    and the split between the two branches gives chi.

    Pick your working readout power a few dB BELOW where the resonance starts
    to move, and use the low-power frequency.

    The frequency axis is always a software sweep, because the ADC has to
    follow the tone. The gain axis depends on `gain_spacing`:

      'log'     software sweep. Punch out spans tens of dB and the interesting
                branch is at LOW power, so log spacing is what you want for a
                first measurement. Costs one program upload per grid point.
      'linear'  HARDWARE loop. pulse.gain is a SweepableParameter, so the tProc
                can step it, and the uploads drop from f_points * gain_points
                to f_points - typically tens of times faster. QickSweep1D is
                linear only, which is the price.

    Use 'log' to find the transition, then 'linear' over a narrow gain range
    around it when you want speed or a finer grid.

    Args:
        rfsoc: RFSoC instance
        resonator: Element instance, must have an ADC
        f_start: start frequency (Hz)
        f_stop: stop frequency (Hz)
        f_points: number of frequency points
        gain_start: lowest readout gain (must be > 0 for log spacing)
        gain_stop: highest readout gain (<= 1)
        gain_points: number of gain points
        gain_spacing: 'log' (default, software sweep) or 'linear' (hardware
            loop, much faster). See the note above.
        pulse_name: name of the readout pulse
        hard_avg: hardware averages, averaged on the FPGA
        soft_avg: software averages, the whole map repeated and averaged
        final_delay: delay between shots (s)
        plot: whether to plot the 2D map
        verbose: whether to print the scan settings and the extracted branches

    Returns:
        frequencies (Hz), gains, complex IQ with shape (gain_points, f_points)
    """
    if resonator.adc is None:
        raise ValueError(f"element '{resonator.name}' has no ADC, it cannot be read out")
    if f_points < 2 or gain_points < 2:
        raise ValueError(f"need at least 2 points per axis, got {f_points} x {gain_points}")
    if f_stop <= f_start:
        raise ValueError(f"f_stop ({f_stop:.4e}) must be above f_start ({f_start:.4e})")
    if not 0 < gain_stop <= 1:
        raise ValueError(f"gain_stop must be in (0, 1], got {gain_stop}")

    if gain_spacing == 'log':
        if gain_start <= 0:
            raise ValueError(f"log spacing needs gain_start > 0, got {gain_start}")
        gains = np.logspace(np.log10(gain_start), np.log10(gain_stop), gain_points)
    elif gain_spacing == 'linear':
        gains = np.linspace(gain_start, gain_stop, gain_points)
    else:
        raise ValueError(f"gain_spacing must be 'log' or 'linear', got '{gain_spacing}'")

    rfsoc.display(f"Running punchout\nDAC:{resonator.dac.channel_num} -> ADC:{resonator.adc.channel_num}")

    readout_pulse = resonator.pulses[pulse_name]
    readout_adc = resonator.adc

    gain_in_hardware = (gain_spacing == 'linear')

    total = f_points * gain_points
    shot = readout_pulse.length.get() + final_delay
    # every program upload costs ~100 ms. A hardware gain loop uploads once per
    # frequency; a software one uploads once per grid point.
    uploads = f_points if gain_in_hardware else total
    seconds = total * hard_avg * soft_avg * shot + uploads * 0.1
    if verbose:
        print(f"resonator        : {resonator.name} (DAC {resonator.dac.channel_num}, ADC {readout_adc.channel_num})")
        print(f"span             : {f_start/1e9:.6f} - {f_stop/1e9:.6f} GHz, {f_points} points")
        print(f"gain             : {gains[0]:.4g} - {gains[-1]:.4g}, {gain_points} points ({gain_spacing})")
        print(f"                   = {20*np.log10(gains[-1]/gains[0]):.0f} dB of range")
        print(f"gain sweep       : {'hardware loop' if gain_in_hardware else 'software sweep'}")
        print(f"grid             : {total} points, {hard_avg} hardware x {soft_avg} software")
        print(f"uploads          : {uploads}")
        print(f"estimated time   : {seconds/60:.1f} min")

    # gain is a scan setting, and the sweep parks every parameter at its last
    # value, so put all three back afterwards
    with restored(readout_pulse.freq, readout_adc.freq, readout_pulse.gain):
        # NOTE: restored() snapshots gain BEFORE the QickSweep1D is assigned, so
        # the plain number is what comes back. Putting a number back also drops
        # the parameter out of qi.swept_params.
        software_sweeps = [
            SoftwareSweep([readout_pulse.freq, readout_adc.freq],
                          f_start, f_stop, f_points),
        ]
        hardware_loop_counts = {}

        if gain_in_hardware:
            readout_pulse.gain.set(QickSweep1D("gain", gain_start, gain_stop))
            hardware_loop_counts = {"gain": gain_points}
        else:
            # list order is outer -> inner: one frequency trace per power.
            # gain takes an explicit array, which is how log spacing gets in.
            software_sweeps.insert(0, SoftwareSweep(readout_pulse.gain, gains))

        run_config = RunConfig(
            measurement_name=f"punchout_{resonator.name}_{f_start:.4e}_{f_stop:.4e}",
            # one experiment per sample, so a cooldown's runs stay together
            experiment_name=f"{resonator.name}",
            sample_name=f"{resonator.name}",
            acquisition_mode='accumulated',
            n_shots=hard_avg,
            soft_avgs=soft_avg,
            final_delay=final_delay,
            software_sweeps=software_sweeps,
            hardware_loop_counts=hardware_loop_counts,
        )

        run_id = rfsoc.run(readout(resonator), run_config)

    # ---------------- recover ----------------
    dataset = load_by_id(run_id)
    data = dataset.get_parameter_data()

    name = list(data)[0]                  # 'iq', or 'iq_ch2' with several ADCs
    block = data[name]
    iq_flat = np.asarray(block[name]).ravel()

    # Both mechanisms register the swept parameter as a dataset setpoint, so
    # the recovery below works whether gain was a hardware loop or a software
    # sweep: the grid is rebuilt from the columns, not from the acquisition
    # order. Only parameters[0] of a SoftwareSweep is stored, so the columns
    # are the pulse gain and the pulse frequency.
    gain_key, freq_key = readout_pulse.gain.full_name, readout_pulse.freq.full_name
    for key in (gain_key, freq_key):
        if key not in block:
            raise KeyError(f"expected column '{key}' in the dataset, found {list(block)}")
    gain_flat = np.asarray(block[gain_key]).ravel()
    freq_flat = np.asarray(block[freq_key]).ravel()

    # sort into a grid with frequency varying fastest, then fold
    order = np.lexsort((freq_flat, gain_flat))
    freqs = np.unique(freq_flat)
    gains = np.unique(gain_flat)
    iq = iq_flat[order].reshape(len(gains), len(freqs))

    resonances = find_resonances(freqs, iq)
    amplitude = np.abs(iq)

    # what the run CONCLUDED, next to the raw map: the two branches and the
    # shift between them are the result of a punch out
    dataset.add_metadata("f_res_low_power", float(resonances[0]))
    dataset.add_metadata("f_res_high_power", float(resonances[-1]))
    dataset.add_metadata("shift", float(resonances[-1] - resonances[0]))
    dataset.add_metadata("gain_spacing", gain_spacing)

    if verbose:
        low, high = resonances[0], resonances[-1]
        print(f"stored as run    : {run_id}")
        print(f"low power f_res  : {low/1e9:.6f} GHz  (gain {gains[0]:.4g})")
        print(f"high power f_res : {high/1e9:.6f} GHz  (gain {gains[-1]:.4g})")
        print(f"shift            : {(high-low)/1e6:+.3f} MHz")
        if abs(high - low) < 2 * (freqs[1] - freqs[0]):
            print("WARNING: the two branches differ by less than two frequency steps.")
            print("         Widen the gain range, or the qubit may not be coupled.")

    if plot:
        plot_punchout(freqs, gains, iq, resonances,
                      log_gain=(gain_spacing == 'log'),
                      title=f"{dataset.name} (run {run_id})")

    rfsoc.display_ready()
    
    return freqs, gains, iq, resonances
