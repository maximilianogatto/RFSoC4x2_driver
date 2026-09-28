"""What qickodes does not give you for sweeps.

`SoftwareSweep` and `QickSweep1D` already work, so they are used directly at
the call site rather than wrapped - a sweep should say what it moves.

What is missing is putting the parameters back: qickodes leaves every swept
parameter at its LAST value, so without `restored` the next measurement
inherits the end of the previous scan.
"""

from contextlib import contextmanager


@contextmanager
def restored(*parameters):
    """Put parameters back where they were once the block finishes.

    Snapshots the CURRENT value of each parameter, not the configured default,
    so anything hand-tuned earlier in the session survives.

    This restores the software state only: a qcodes parameter lives in python,
    and the value reaches the board on the next `run()`, when the program is
    compiled. The point is that the next program is built from the right
    numbers.

    Parameters
    ----------
    *parameters
        The qcodes parameters to snapshot and restore.

    Examples
    --------
    >>> with restored(readout_pulse.freq, resonator.adc.freq):
    ...     rfsoc.run(macros, run_config)
    """
    saved = [(parameter, parameter.get()) for parameter in parameters]
    try:
        yield
    finally:
        # runs even if the measurement raised, which is the whole point
        for parameter, value in saved:
            parameter.set(value)
