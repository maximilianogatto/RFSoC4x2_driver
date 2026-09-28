import numpy as np
import matplotlib.pyplot as plt

from qcodes.dataset import (load_by_id)

from ..rfsoc import RFSoC, RunConfig
from ..elements import Element
from ..sequences import readout

def detect_arrival(t, iq, pulse_length, smooth=7, min_snr=10):
    """Pulse arrival time (s) in a decimated trace, measured from the ADC trigger."""
    t = np.ravel(t)
    amp = np.convolve(np.abs(np.ravel(iq)), np.ones(smooth) / smooth, mode="same")
    # split the samples into "no signal" and "signal", threshold halfway between them
    thr = 0.5 * (amp.min() + amp.max())
    for _ in range(100):
        low, high = amp[amp <= thr].mean(), amp[amp > thr].mean()
        thr, old = 0.5 * (low + high), thr
        if abs(thr - old) < 1e-9 * high:
            break

    noise = amp[amp <= thr].std()
    if noise == 0:
        raise RuntimeError("flat trace: no signal and no noise, check the wiring")
    snr = (high - low) / noise
    above = amp > thr
    if snr < min_snr:
        raise RuntimeError(f"no clear pulse (S/N {snr:.1f}): average more")
    if above[:smooth].any():
        raise RuntimeError("pulse already there when the ADC starts: use a longer window")
    if above[-smooth:].any():
        raise RuntimeError("pulse still on when the window ends: use a longer window")
    i = int(np.argmax(above))                         # first sample above the threshold
    j = len(above) - 1 - int(np.argmax(above[::-1]))  # last sample above the threshold
    if i == 0:
        raise RuntimeError("edge on the first sample, cannot interpolate")
    arrival = np.interp(thr, amp[i - 1:i + 1], t[i - 1:i + 1])  # between the two samples
    width = t[j] - t[i]
    if abs(width - pulse_length) > 0.2 * pulse_length:
        raise RuntimeError(f"pulse looks {width*1e9:.0f} ns long, expected {pulse_length*1e9:.0f} ns")
    return arrival


def tof_calibration(rfsoc: RFSoC, resonator: Element, pulse_length: float, window: float | None = None, margin: float = 20e-9, smooth=7, min_snr=10, soft_avg = 100, final_delay: float = 5e-6, verbose = True, plot = True) -> tuple[float, int]:
    """Autocalibrate the time-of-flight delay for a resonator.

    The window opens at t = 0 (the configured time_of_flight is deliberately
    ignored) and has to be long enough to hold the round trip plus the pulse:
    opening it at the value we are trying to measure would assume the answer.

    Args:
        rfsoc: RFSoC instance
        resonator: Element instance, must have an ADC
        pulse_length: expected pulse length (s)
        window: acquisition window (s). None picks one wide enough on its own.
        margin: margin to add to the detected arrival time (s)
        smooth: smoothing length for arrival detection (samples)
        min_snr: minimum S/N ratio for arrival detection
        soft_avg: number of soft averages to take
        final_delay: delay between shots (s). Only the resonator has to
            ring down here, so a few microseconds is plenty.
        verbose: whether to print verbose output
        plot: whether to plot the results
    Returns:
        arrival + margin: detected arrival time + margin (s)
        run_id: QCoDeS run ID of the measurement
    """
    if resonator.adc is None:
        raise ValueError(f"element '{resonator.name}' has no ADC, there is nothing to calibrate")

    original_window = resonator.adc.length.get()
    capped = False
    if window is None:
        # the delay is what we are measuring, so leave generous room for it.
        # coax is ~5 ns/m, so even a long fridge run plus amplifier group delay
        # is well under a microsecond.
        window = pulse_length + 2e-6

        # ...but the decimated buffer may not hold that much. Cap it at the
        # longest window that fits, rather than raising: a shorter window still
        # finds the pulse as long as the pulse arrives inside it.
        wanted = resonator.window_samples(window)
        if wanted > resonator.buf_maxlen:
            window = window * resonator.buf_maxlen / wanted
            capped = True

    if window <= pulse_length:
        raise ValueError(
            f"window {window*1e9:.0f} ns must be longer than the pulse "
            f"{pulse_length*1e9:.0f} ns plus the expected delay"
        )

    # Buffer budget: the DECIMATED buffer holds n_shots WINDOWS, not pulses.
    # A wider window therefore costs hardware averaging, which is why soft_avg
    # does most of the work in this calibration.
    samples = resonator.window_samples(window)
    max_shots = resonator.buf_maxlen // samples
    if max_shots < 1:
        raise ValueError(
            f"a window of {samples} samples does not fit in the decimated buffer of "
            f"{resonator.buf_maxlen} samples. Shorten it, or use acquisition_mode='ddr4'."
        )
    sample_period = window / samples

    if verbose:
        print(f"decimated buffer : {resonator.buf_maxlen} samples")
        print(f"window           : {window*1e9:.0f} ns -> {samples} samples ({sample_period*1e9:.2f} ns each)")
        print(f"averaging        : {max_shots} hardware x {soft_avg} software")
        seconds = max_shots * soft_avg * (window + final_delay) + 0.1
        print(f"estimated time   : {seconds:.1f} s")
        if capped:
            print(f"NOTE: window capped by the {resonator.buf_maxlen}-sample buffer. "
                  f"If the pulse falls outside it, use acquisition_mode='ddr4'.")
        if margin < 3 * sample_period:
            print(f"WARNING: margin {margin*1e9:.1f} ns is less than three samples")
        print(f"Running autocalibration for {resonator.name} at {resonator.frequency} Hz.")

    try:
        resonator.adc.length.set(window)
        readout_sequence = readout(resonator, time_of_flight=0, t_delay=10e-9)

        run_config = RunConfig( measurement_name = f"autocalibrate_tof_{resonator.name}",
                                experiment_name = f"autocalibrate_tof_{resonator.name}",
                                sample_name = f"{resonator.name}",
                                acquisition_mode = 'decimated',
                                n_shots = max_shots,
                                soft_avgs = soft_avg,
                                final_delay = final_delay)

        run_id = rfsoc.run(readout_sequence, run_config)
    finally:
        # always put the window back, even if the run or the fit fails
        resonator.adc.length.set(original_window)

    dataset = load_by_id(run_id)
    data = dataset.get_parameter_data()

    name = list(data)[0]
    block = data[name]

    iq = np.asarray(block[name]).ravel()
    time = np.asarray(block['time']).ravel()

    arrival = detect_arrival(time, iq, pulse_length, smooth=smooth, min_snr=min_snr)
    tof = arrival + margin

    # The run's own snapshot holds the OLD time_of_flight, because it was taken
    # before the fit. Attach the result to this dataset so the calibration can
    # be looked up later: dataset.metadata['time_of_flight'].
    dataset.add_metadata("time_of_flight", tof)
    dataset.add_metadata("tof_arrival", arrival)
    dataset.add_metadata("tof_margin", margin)

    if verbose:
        print(f"arrival {arrival*1e9:.2f} ns + margin {margin*1e9:.2f} ns = {tof*1e9:.2f} ns "
              f"(was {resonator.time_of_flight*1e9:.2f} ns)")

    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax[0].plot(time * 1e9, np.abs(iq), label="Amplitude")
        ax[0].axvline(arrival * 1e9, color='r', linestyle='--', label=f"Arrival: {arrival*1e9:.2f} ns")
        ax[0].axvline(tof * 1e9, color='g', linestyle='--', label=f"Arrival + margin: {tof*1e9:.2f} ns")
        ax[0].set_ylabel("Amplitude")
        ax[0].legend()
        ax[1].plot(time * 1e9, np.rad2deg(np.unwrap(np.angle(iq))), label="Phase")
        ax[1].set_ylabel("Phase [degrees]")
        ax[1].set_xlabel("Time [ns]")
        fig.suptitle(f"{dataset.name} (run {run_id})")

    rfsoc.update_spec([('time_of_flight', tof)], element=resonator.name)
    print(f"Resonator {resonator.name} spec updated with time-of-flight delay: {tof*1e9:.2f} ns")

    return tof, run_id
