"""A QM-style configuration layer over qickodes for the RFSoC 4x2.

The idea is the one Quantum Machines uses: describe the setup as *elements*
(a qubit, a resonator) that own named *pulses*, then write measurements in
terms of those names instead of DAC and ADC numbers.

Layers, from data to hardware:

    specs.py      pure dataclasses. No qickodes import, no board needed.
    config.py     a Config holding specs, with validation.
    build.py      specs -> live qickodes objects (singledispatch, one builder
                  per type).
    elements.py   the live Element: its DAC, its ADC, its pulses.
    sequences.py  small fragments that return lists of macros.
    rfsoc.py      RFSoC: the connection, the elements, and run().

Typical use::

    from rfsoc4x2 import RFSoC, RunConfig, Config, ResonatorSpec, ConstantPulseSpec
    from rfsoc4x2.sequences import readout

    config = Config(
        elements=[ResonatorSpec(name="r0", dac=0, adc=0, frequency=1e9,
                                readout_length=10e-6, time_of_flight=0.0)],
        pulses=[ConstantPulseSpec(name="readout", element="r0",
                                  length=10e-6, gain=0.5)],
    )
    rf = RFSoC("qi", "192.168.1.10", station, "data.db", config, ns_port=8000)
    rf.run(readout(rf.element("r0")),
           RunConfig(measurement_name="loopback",
                     experiment_name="commissioning", sample_name="none"))

`sweeps.py` and `programs.py` are placeholders for the sweep layer and the
measurement library (punch-out, Rabi, T1). They are not written yet.
"""

from .config import Config
from .elements import Element
from .rfsoc import RFSoC, RunConfig
from .specs import (
    ArbitraryPulseSpec,
    ConstantPulseSpec,
    CorrectedConstantPulseSpec,
    ElementSpec,
    EnvelopeSpec,
    FlatTopPulseSpec,
    GaussianDragEnvelopeSpec,
    GaussianEnvelopeSpec,
    MuxedConstantPulseSpec,
    PulseSpec,
    QubitSpec,
    ResonatorSpec,
)

__version__ = "0.1.0"

__all__ = [
    # live
    "RFSoC",
    "RunConfig",
    "Config",
    "Element",
    # element specs
    "ElementSpec",
    "QubitSpec",
    "ResonatorSpec",
    # pulse specs
    "PulseSpec",
    "ConstantPulseSpec",
    "CorrectedConstantPulseSpec",
    "ArbitraryPulseSpec",
    "FlatTopPulseSpec",
    "MuxedConstantPulseSpec",
    # envelope specs
    "EnvelopeSpec",
    "GaussianEnvelopeSpec",
    "GaussianDragEnvelopeSpec",
    "__version__",
]
