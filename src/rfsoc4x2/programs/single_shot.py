import math

import numpy as np
import matplotlib.pyplot as plt

from qcodes.dataset import (load_by_id)

from ..rfsoc import RFSoC, RunConfig
from ..elements import Element
from ..sequences import play, readout
from ..sweeps import restored

# hardware sweeps
from qick.asm_v2 import QickSweep1D


def to_probability(iq, ground: complex, excited: complex) -> np.ndarray:
    """Excited-state population from an AVERAGED IQ point.

    The measurement is linear in the state, so an average over shots lands on
    the line between the two blob centres:

        <IQ> = (1 - P1) * ground + P1 * excited

    P1 is therefore the fractional distance along that line. Values slightly
    outside 0..1 are normal - they are noise on the projection, not an error.

    Args:
        iq: averaged IQ, a scalar or an array, from an 'accumulated' run.
        ground, excited: blob centres from `single_shot_readout`.

    Returns:
        the excited-state population, same shape as `iq`.
    """
    axis = excited - ground
    return np.real((np.asarray(iq) - ground) * np.conj(axis)) / abs(axis) ** 2


def _robust_sigma(values: np.ndarray) -> float:
    """Width of the core of a distribution, ignoring tails.

    The median absolute deviation, scaled by 1.4826 so it equals the standard
    deviation for a gaussian.
    """
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def _best_threshold(ground_proj: np.ndarray, excited_proj: np.ndarray) -> tuple[float, float, float]:
    """The threshold that separates the two clouds best, and the error rates.

    Scans every midpoint between neighbouring samples rather than assuming the
    clouds are gaussian or equally wide, which they are not: T1 decay during
    the readout window drags part of the excited cloud towards ground.
    """
    candidates = np.unique(np.concatenate([ground_proj, excited_proj]))
    ground_above = np.searchsorted(np.sort(ground_proj), candidates, side='right')
    excited_above = np.searchsorted(np.sort(excited_proj), candidates, side='right')

    # fraction of each cloud landing on the wrong side of the threshold
    p_ground_wrong = 1 - ground_above / len(ground_proj)
    p_excited_wrong = excited_above / len(excited_proj)
    fidelity = 1 - (p_ground_wrong + p_excited_wrong) / 2

    best = int(np.argmax(fidelity))
    return float(candidates[best]), float(p_ground_wrong[best]), float(p_excited_wrong[best])


def single_shot_readout(rfsoc: RFSoC, qubit: Element, resonator: Element | None = None, pi_gain: float | None = None, pulse_name: str = 'pi', shots: int = 10000, final_delay: float = 200e-6, plot: bool = True, verbose: bool = True) -> dict:
    """Measure the two IQ blobs, and with them the readout calibration.

    Runs `shots` single-shot readouts with the qubit in |0>, and `shots` with
    it in |1>, and reports where the two clouds sit, how well a threshold
    separates them, and the angle of the line between them.

    Both states come from ONE run: the drive gain is a two-point hardware
    sweep, 0 then `pi_gain`, so the clouds are taken under identical conditions
    and interleaved in time. Drift affects both equally.

    What this gives you:

      ground, excited   the reference points that turn any averaged IQ into a
                        population, via `to_probability`
      angle             the readout axis, set by the cable lengths
      threshold         for state assignment in 'state population' mode
      fidelity          how good the readout is, before anything else is blamed

    Args:
        rfsoc: RFSoC instance
        qubit: the element to drive
        resonator: the element to read out. Defaults to the resonator named
            in the qubit's spec (QubitSpec.readout).
        pi_gain: the pi pulse gain. Defaults to the pulse's current gain, which
            is the calibrated one once rabi's result went through update_spec.
        pulse_name: which of the qubit's pulses to drive with
        shots: single shots per state
        final_delay: delay between shots (s). At least 5x T1: an unrelaxed
            qubit puts ground-state shots into the excited cloud.
        plot: whether to plot the clouds and the histogram
        verbose: whether to print the result

    Returns:
        dict with ground, excited, angle, threshold, fidelity, p_e_given_g,
        p_g_given_e, separation, snr, fidelity_limit
    """
    resonator = qubit.resolve_readout(resonator)
    if resonator.adc is None:
        raise ValueError(f"element '{resonator.name}' has no ADC, it cannot be read out")

    drive = qubit.pulses[pulse_name]
    if pi_gain is None:
        pi_gain = drive.gain.get()
    if not 0 < pi_gain <= 1:
        raise ValueError(f"pi_gain must be in (0, 1], got {pi_gain}. Run rabi() first.")

    # the accumulated buffer has to hold every shot of both states, because
    # 'accumulated shots' keeps them instead of averaging on the FPGA
    capacity = rfsoc.qi.soccfg["readouts"][resonator.adc.channel_num].get("avg_maxlen")
    if capacity is not None and shots * 2 > capacity:
        raise ValueError(
            f"{shots} shots per state needs {shots*2} slots but the accumulated "
            f"buffer holds {capacity}. Use shots <= {capacity // 2}."
        )

    shot_time = resonator.adc.length.get() + final_delay
    if verbose:
        print(f"qubit            : {qubit.name}, pulse '{pulse_name}' at gain {pi_gain:.4g}")
        print(f"readout          : {resonator.name}, window {resonator.adc.length.get()*1e9:.0f} ns")
        print(f"shots            : {shots} per state, one hardware loop of 2")
        if capacity is not None:
            print(f"buffer           : {shots*2} / {capacity} slots")
        print(f"estimated time   : {2*shots*shot_time/60:.1f} min")

    with restored(drive.gain):
        # two points: |0> at gain 0, |1> at the pi gain. One run, interleaved.
        drive.gain.set(QickSweep1D("gain", 0.0, pi_gain))

        run_config = RunConfig(
            measurement_name=f"single_shot_{qubit.name}",
            experiment_name=f"{qubit.name}",
            sample_name=f"{qubit.name}",
            acquisition_mode='accumulated shots',   # keeps every shot
            n_shots=shots,
            soft_avgs=1,                            # qickodes requires it here
            final_delay=final_delay,
            hardware_loop_counts={"gain": 2},
        )

        macros = play(qubit, pulse_name) + readout(resonator)
        run_id = rfsoc.run(macros, run_config)

    # ---------------- recover ----------------
    dataset = load_by_id(run_id)
    data = dataset.get_parameter_data()

    name = list(data)[0]
    block = data[name]
    iq = np.asarray(block[name]).ravel()
    gain = np.asarray(block[drive.gain.full_name]).ravel()

    # split by the gain column rather than reshaping: no assumption about the
    # order the shot and gain axes come back in
    low = gain < (gain.min() + gain.max()) / 2
    iq_ground, iq_excited = iq[low], iq[~low]

    centre_ground, centre_excited = iq_ground.mean(), iq_excited.mean()
    axis = centre_excited - centre_ground
    angle = float(np.angle(axis))
    separation = float(abs(axis))

    # project both clouds onto the line between the centres
    rotate = np.exp(-1j * angle)
    ground_proj = np.real(iq_ground * rotate)
    excited_proj = np.real(iq_excited * rotate)

    threshold, p_ground_wrong, p_excited_wrong = _best_threshold(ground_proj, excited_proj)
    fidelity = 1 - (p_ground_wrong + p_excited_wrong) / 2
    # SNR from the CORE of each blob. The excited cloud has a tail from T1
    # decay during the readout; a plain std would count that tail as noise and
    # a plain mean would pull the centre towards ground, both lowering the SNR.
    # Medians and the median absolute deviation ignore the tail.
    core_separation = float(abs(np.median(excited_proj) - np.median(ground_proj)))
    noise = (_robust_sigma(ground_proj) + _robust_sigma(excited_proj)) / 2
    snr = core_separation / noise if noise > 0 else np.inf
    # The best fidelity that noise alone would allow, if both blobs were clean
    # gaussians. A measured fidelity well below it means shots are landing in
    # the wrong blob for another reason: T1 in the readout, an incomplete pi
    # pulse, or a qubit not fully in |0> at the start.
    fidelity_limit = 1 - 0.5 * math.erfc(snr / (2 * math.sqrt(2)))

    result = {
        "ground": complex(centre_ground),
        "excited": complex(centre_excited),
        "angle": angle,
        "threshold": threshold,
        "fidelity": fidelity,
        "p_e_given_g": p_ground_wrong,     # a |0> shot read as 1
        "p_g_given_e": p_excited_wrong,    # a |1> shot read as 0, mostly T1
        "separation": separation,
        "snr": snr,
        "fidelity_limit": fidelity_limit,
        "run_id": run_id,
    }

    for key, value in result.items():
        if key != "run_id":
            dataset.add_metadata(key, str(value) if isinstance(value, complex) else value)

    if verbose:
        print(f"stored as run    : {run_id}")
        print(f"ground           : {centre_ground:.5g}")
        print(f"excited          : {centre_excited:.5g}")
        print(f"separation / snr : {separation:.5g} / {snr:.2f}")
        print(f"fidelity         : {fidelity*100:.2f}%  "
              f"(P(e|g) {p_ground_wrong*100:.2f}%, P(g|e) {p_excited_wrong*100:.2f}%)")
        print(f"noise limit      : {fidelity_limit*100:.2f}%  (what the SNR alone would allow)")
        if fidelity_limit - fidelity > 0.02:
            print("NOTE: the fidelity is well below the noise limit, so noise is not what")
            print("      limits it: more averaging or a longer readout will not help.")
        if p_excited_wrong > 3 * p_ground_wrong and p_excited_wrong > 0.02:
            print("NOTE: the excited cloud leaks into ground far more than the reverse.")
            print("      Usually T1 decay during the readout window: shorten it,")
            print("      or the pi pulse is not fully inverting.")

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(13, 6))
        sub = slice(None, None, max(1, len(iq_ground) // 5000))   # keep the plot light
        ax[0].plot(iq_ground[sub].real, iq_ground[sub].imag, '.', ms=1, alpha=0.3, label="|0>")
        ax[0].plot(iq_excited[sub].real, iq_excited[sub].imag, '.', ms=1, alpha=0.3, label="|1>")
        ax[0].plot([centre_ground.real, centre_excited.real],
                   [centre_ground.imag, centre_excited.imag], 'k.-', ms=10, label="centres")
        ax[0].set_xlabel("I [ADC units]")
        ax[0].set_ylabel("Q [ADC units]")
        ax[0].set_title("Single-shot IQ")
        ax[0].set_aspect('equal')
        ax[0].legend(markerscale=8)

        bins = np.linspace(min(ground_proj.min(), excited_proj.min()),
                           max(ground_proj.max(), excited_proj.max()), 120)
        ax[1].hist(ground_proj, bins=bins, alpha=0.6, label="|0>")
        ax[1].hist(excited_proj, bins=bins, alpha=0.6, label="|1>")
        ax[1].axvline(threshold, color='k', ls='--', label=f"threshold (F={fidelity*100:.1f}%)")
        ax[1].set_xlabel("projection onto the readout axis")
        ax[1].set_ylabel("shots")
        ax[1].set_title("Separation")
        ax[1].legend()

        fig.suptitle(f"{dataset.name} (run {run_id})")
        plt.tight_layout()

    return result
