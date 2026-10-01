import numpy as np
import matplotlib.pyplot as plt

from qcodes.dataset import (load_by_id)

from ..rfsoc import RFSoC, RunConfig
from ..elements import Element
from ..sweeps import restored
from .resonator_spectroscopy_hw import MIN_CONFIG_TO_TRIGGER

# software sweeps: the gain
from qickodes.instrument_v2 import SoftwareSweep

# hardware sweeps: the frequency. The readout tone and the ADC down-conversion
# frequency are swept together by the tProc (ReadoutWindow + ConfigReadout).
from qick.asm_v2 import QickSweep1D
from qickodes.macros_v2 import ConfigReadout, DelayAuto, PlayPulse, Trigger
from qickodes.readout_window_v2 import ReadoutWindow


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
    # normalised = amplitude / np.median(amplitude, axis=1, keepdims=True)
    scale = 'log' if log_gain else 'linear'

    fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    mesh = ax[0].pcolormesh(freqs / 1e9, gains, amplitude, shading='nearest', cmap='seismic')
    #ax[0].plot(resonances / 1e9, gains, 'r.-', lw=1, ms=4, label="resonance")
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

    The frequency axis is a HARDWARE sweep: the tProc steps the readout tone
    and the ADC down-conversion frequency together. The gain axis depends on
    `gain_spacing`:

      'log'     points equally spaced in dB. Punch out spans tens of dB and the
                interesting branch is at LOW power, so this is what you want for
                a first measurement. QickSweep1D is linear only, so the gain is
                a software sweep: one program (about 1 s) per power.
      'linear'  points equally spaced in gain, for a fine grid over a narrow
                range around the transition. The gain is a hardware loop too,
                so the whole map is ONE program, as long as gain_points *
                f_points fits in the ADC's accumulation buffer (16384 points).
                If it does not, the gain falls back to a software sweep.

    The time is dominated by the integration, gain_points * f_points *
    hard_avg * (pulse + final_delay): the hardware sweeps remove the program
    uploads, not the averaging.

    Args:
        rfsoc: RFSoC instance
        resonator: Element instance, must have an ADC
        f_start: start frequency (Hz)
        f_stop: stop frequency (Hz)
        f_points: number of frequency points
        gain_start: lowest readout gain (must be > 0 for log spacing)
        gain_stop: highest readout gain (<= 1)
        gain_points: number of gain points
        gain_spacing: 'log' (default) or 'linear'. See the note above.
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

    total = f_points * gain_points
    # QickSweep1D is linear, so only a linear gain grid can be a hardware loop, and
    # the whole grid has to fit in the ADC's accumulation buffer. Then the entire
    # map is ONE program. Otherwise the gain is a software sweep: one program per
    # power, each with the frequency sweep in hardware.
    gain_in_hardware = (gain_spacing == 'linear') and total <= avg_maxlen
    shot = readout_pulse.length.get() + final_delay
    # building and uploading a program takes about a second
    uploads = 1 if gain_in_hardware else gain_points
    seconds = total * hard_avg * soft_avg * shot + uploads * 1.0
    if verbose:
        print(f"resonator        : {resonator.name} (DAC {resonator.dac.channel_num}, ADC {readout_adc.channel_num})")
        print(f"span             : {f_start/1e9:.6f} - {f_stop/1e9:.6f} GHz, {f_points} points")
        print(f"gain             : {gains[0]:.4g} - {gains[-1]:.4g}, {gain_points} points ({gain_spacing})")
        print(f"                   = {20*np.log10(gains[-1]/gains[0]):.0f} dB of range")
        print(f"frequency sweep  : hardware")
        print(f"gain sweep       : {'hardware (2D: the whole map is one program)' if gain_in_hardware else 'software (one program per power)'}")
        if gain_spacing == 'linear' and not gain_in_hardware:
            print(f"                   the {total}-point grid does not fit in the {avg_maxlen}-point accumulation buffer")
        print(f"grid             : {total} points, {hard_avg} hardware x {soft_avg} software")
        print(f"uploads          : {uploads}")
        print(f"estimated time   : {seconds/60:.1f} min")

    # The readout frequency lives on a ReadoutWindow, which the ConfigReadout
    # macro sends to the ADC inside the loop. The pulse and the window use the
    # SAME sweep name and range, so the tProc moves them together.
    window = ReadoutWindow(readout_adc, f"{readout_adc.short_name}_sweep")

    # gain is a scan setting, and the sweep parks every parameter at its last
    # value, so put all three back afterwards
    with restored(readout_pulse.freq, readout_adc.freq, readout_pulse.gain):
        # NOTE: restored() snapshots freq BEFORE the QickSweep1D is assigned, so
        # the plain number is what comes back. Putting a number back also drops
        # the parameter out of qi.swept_params.
        try:
            readout_pulse.freq.set(QickSweep1D("freq", f_start, f_stop))
            window.freq.set(QickSweep1D("freq", f_start, f_stop))

            # list order is outer -> inner: one frequency trace per power
            if gain_in_hardware:
                readout_pulse.gain.set(QickSweep1D("gain", gain_start, gain_stop))
                software_sweeps = []
                hardware_loop_counts = {"gain": gain_points, "freq": f_points}
            else:
                # gain takes an explicit array, which is how log spacing gets in
                software_sweeps = [SoftwareSweep(readout_pulse.gain, gains)]
                hardware_loop_counts = {"freq": f_points}

            macros = [
                DelayAuto(qi, t=10e-9),
                ConfigReadout(qi, window, t=0),
                Trigger(qi, readout_adc, t=resonator.time_of_flight),
                PlayPulse(qi, readout_pulse),
                DelayAuto(qi, t=10e-9),
            ]

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
    iq_flat = np.asarray(block[name]).ravel()

    # Both kinds of sweep register the swept parameter as a dataset setpoint, so
    # the grid is rebuilt from the columns, not from the acquisition order: the
    # pulse gain (software) and the pulse frequency (hardware). The ADC
    # frequency is stored too, as a second column of the same grid.
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
    dataset.add_metadata("sweep", "hardware frequency and gain" if gain_in_hardware
                         else "hardware frequency, software gain")

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
