# rfsoc4x2

A QM-style configuration layer over [qickodes](https://github.com/aalto-qcd/qickodes)
for the RFSoC 4x2.

Describe the setup as **elements** (a qubit, a resonator) that own named
**pulses**, then write measurements in terms of those names instead of DAC and
ADC numbers. The layer also does the bookkeeping that is easy to forget: the
ADC frequency follows the resonator frequency, the DAC/ADC pair is matched, and
a qubit never gets an ADC.

## Requirements

- tProc **v2** firmware on the board (`v2r27` is what this was written against).
  The v2 API refuses to connect to v1 firmware.
- Python **3.10+** (qickodes itself needs 3.9+, but this package uses
  `X | None` annotations).

## Install on the lab computer

```bash
git clone --recurse-submodules https://github.com/maximilianogatto/PhD.git
cd PhD
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# qickodes from the local checkout, so local patches are picked up
pip install -e "src/libraries/qickodes"

# this package
pip install -e "src/drivers/RFSoC 4x2"
```

Check it imports:

```bash
python -c "import rfsoc4x2; print(rfsoc4x2.__version__)"
```

> The folder name contains a space, so keep the quotes around the path.

If the repository is already cloned, pull the submodule instead:

```bash
git submodule update --init "src/drivers/RFSoC 4x2"
```

## Board setup (do this once)

On the board, edit `pyro4/pyro_service.py`:

```python
bitfile    = '/path/to/your/tprocv2/qick_4x2.bit'   # .hwh with the same name beside it
proxy_name = 'myqick'      # qickodes looks for this exact name
ns_port    = 8000
```

`proxy_name` matters: qickodes calls `make_proxy(host, port)` without a name,
so it looks for the default `myqick`. A server registered as `rfsoc` will not
be found.

Start it:

```bash
sudo -s
source /etc/profile
python -m qick.pyro_cli myqick -n localhost -p 8000
```

## A day's work

```python
from qcodes import Station
from rfsoc4x2 import RFSoC, Config
from rfsoc4x2.sequences import readout
from rfsoc4x2.programs.tof_calibration import tof_calibration
from rfsoc4x2.programs.resonator_spectroscopy import resonator_spectroscopy
from rfsoc4x2.programs.punchout import punchout

config = Config.load("setup.json")
rfsoc = RFSoC("qi", "ip.of.board", Station(), "data/rfsoc.db", config, ns_port=8000)
rfsoc.load_calibration("calibration/cal_2026-09-28.json")

freqs, iq = resonator_spectroscopy(rfsoc, rfsoc.element("resonator"),
                                   5.0e9, 5.2e9, 201)
...
rfsoc.save_calibration()        # calibration/cal_<today>.json
```

## Describing a setup

An element owns a DAC, a carrier frequency and (for a resonator) an ADC. A
pulse takes its DAC and carrier from its element, so it only says how far it
sits from that carrier. Units are SI everywhere: Hz, seconds, degrees, gain in
−1…1.

```python
from rfsoc4x2 import (Config, ResonatorSpec, QubitSpec,
                      ConstantPulseSpec, ArbitraryPulseSpec,
                      GaussianDragEnvelopeSpec)

config = Config(
    elements=[
        ResonatorSpec(name="resonator", dac=1, adc=2, frequency=5e9, nqz=2,
                      readout_length=1e-6, time_of_flight=400e-9),
        QubitSpec(name="qubit", dac=0, frequency=3.44e9, nqz=1,
                  readout="resonator"),        # the resonator that reads it out
    ],
    pulses=[
        ConstantPulseSpec(name="readout", element="resonator",
                          length=0.5e-6, gain=1.0),
        ArbitraryPulseSpec(
            name="pi", element="qubit", gain=0.52, detuning=0,
            envelope=GaussianDragEnvelopeSpec(name="ge", sigma=5e-9,
                                              length=20e-9, delta=-161e6,
                                              alpha=-0.12)),
    ],
)
config.save("setup.json")
```

A qubit's `readout` names the resonator coupled to it, so qubit measurements
take the qubit alone — `rabi(rfsoc, rfsoc.element("qubit"))` — and read out
through that resonator. Pass `resonator=` to override it. The name is checked
when the config is built: it must be one of the resonators.

Five pulse types are available: `ConstantPulseSpec` (the readout tone),
`ArbitraryPulseSpec` (a shaped qubit pulse), `CorrectedConstantPulseSpec`,
`FlatTopPulseSpec`, and `MuxedConstantPulseSpec` — the last needs firmware with
a multiplexed generator, which the standard 4x2 image does not have.

## The measurement programs

| Program | Sweeps | Returns |
|---|---|---|
| `tof_calibration` | — (one decimated trace) | `(time_of_flight, run_id)` |
| `resonator_spectroscopy` | frequency | `(freqs, iq)` |
| `punchout` | gain × frequency | `(freqs, gains, iq)` |

Each prints its settings and an **estimated run time** before it starts, writes
its conclusion into the dataset's metadata, and plots the result. Run them in
that order: the time of flight decides where the acquisition window opens,
spectroscopy finds the resonance, and punchout maps it against power.

```python
tof, run_id = tof_calibration(rfsoc, resonator, pulse_length=0.5e-6)
```

`tof_calibration` also writes the value it measured back into the config with
`update_spec`, so every later readout uses it.

## Calibration

Calibrated numbers live at three levels, and `update_spec` reaches all of them.
An attribute that is not on a pulse is looked for on its envelope, so a DRAG
calibration writes to the pulse it belongs to:

```python
rfsoc.update_spec([("time_of_flight", 612e-9)], element="resonator")
rfsoc.update_spec([("gain", 0.52)], pulse="pi")        # the pulse
rfsoc.update_spec([("alpha", -0.31)], pulse="pi")      # its envelope
```

Every change goes into the `Config`, so it lands in the snapshot of every later
run. Save and reload it across sessions:

```python
rfsoc.save_calibration()                                  # cal_<today>.json
rfsoc.load_calibration("calibration/cal_2026-09-28.json")
```

A calibration file holds only what a measurement *measures* — `frequency`,
`gain`, `phase`, `detuning`, `length`, `time_of_flight`, `readout_length`,
`sigma`, `alpha`, `delta` — and no wiring, so yesterday's numbers can be laid
over today's setup. A name the setup does not have raises unless you pass
`strict=False`; a silent skip would mean measuring with a stale value and never
knowing.

A dated directory of these files is also the record of how the sample drifted
through the cooldown.

## Reproducing an old measurement

`RFSoC.run()` puts the whole config into every dataset's snapshot, so:

```python
config = Config.from_run(42)      # the exact setup that produced run 42
```

And what a run *concluded*, as opposed to how it was configured, is in its
metadata:

```python
load_by_id(42).metadata["time_of_flight"]
load_by_id(17).metadata["shift"]          # a punch out
```

## Loopback test

Before the fridge: wire **DAC → 20–30 dB attenuator → ADC**. Do not connect
them directly, the ADC will clip. Start at **500 MHz – 1 GHz**, safely in
Nyquist zone 1 and well inside the balun's 10 MHz – 10 GHz band.

```python
print(rfsoc.qi.soccfg)     # tProc type/revision, sample rates, output pins
rfsoc.run(readout(rfsoc.element("resonator")),
          RunConfig(measurement_name="loopback", experiment_name="commissioning",
                    sample_name="none", acquisition_mode="decimated",
                    n_shots=1, soft_avgs=1))
```

A clean burst in the decimated trace means connection, firmware, config,
builders, elements, sequences and the database all work.

## Layout

| File | Holds |
|---|---|
| `specs.py` | dataclasses only. No qickodes import, no board needed |
| `config.py` | `Config`: validation, save/load, `from_run`, `calibration()` |
| `build.py` | specs → live qickodes objects, one `singledispatch` builder per type |
| `elements.py` | the live `Element`: its DAC, ADC, pulses and readout buffer |
| `sequences.py` | `readout()`, `play()`, `wait()` → lists of macros |
| `sweeps.py` | `restored()`, which puts swept parameters back afterwards |
| `rfsoc.py` | `RFSoC` (connection, elements, calibration, `run()`) and `RunConfig` |
| `programs/` | the measurement routines |

### Design rules

- **Specs are data.** `specs.py` never imports qickodes, so a config can be
  written, checked and version-controlled without a board.
- **`RFSoC` decides what to build and when; `build.py` knows how.** Adding a
  pulse type means one registered builder, nothing else changes.
- **Don't wrap what you don't need to change.** `element.pulse("readout")` *is*
  the qickodes `DacPulse`, so everything in the qickodes docs still applies,
  and `element.qi` reaches the `QickInstrument` through `dac.parent`.
- **A program only adds the sweep.** The config owns the setup; a measurement
  describes what varies.

## Gotchas worth remembering

- **qickodes never puts a swept parameter back** — it is left at the last value
  of the scan. Wrap a run in `restored(...)` or the next measurement inherits
  it. This bites software sweeps too, not just hardware ones.
- **A readout frequency sweep must move the ADC as well.** `AdcChannel.freq` is
  a `ManualParameter`, so it cannot be a hardware loop; sweep
  `[pulse.freq, adc.freq]` together in software. Sweeping only the pulse gives
  a peak at the ADC's frequency and no error at all.
- **A qubit drive frequency *can* be a hardware sweep**, because no ADC has to
  follow it. That is why the chevron example uses `QickSweep1D` and
  spectroscopy does not.
- **`singledispatch` resolves every annotation** in a registered function, not
  just the first argument. Imports used in type hints must be real runtime
  imports, not under `if TYPE_CHECKING`. See the note at the top of `build.py`.
- **Pulse length vs envelope length.** `ConstantPulse.length` is the whole
  pulse; `FlatTopPulse.length` is only the flat part; `ArbitraryPulse` has no
  length at all — its duration is the envelope's.
- **Envelope parameters cannot be hardware-swept** (`sigma`, `delta`, `alpha`
  are `ManualParameter`), but they *can* be changed: the program is recompiled
  from their current values on every run.
- **The decimated buffer is small.** `buf_maxlen // window_samples` is the most
  repetitions that fit, which is why `tof_calibration` leans on `soft_avgs`.
  This limit does **not** apply to `accumulated` mode.
- **`final_delay` should be at least 5×T1**, ideally 10×. Too short biases a
  fitted T1 low, with no warning.

## Still to do

- `sweeps.py` grows a `Sweep` layer when Rabi and T1 need hardware loops: the
  `QickSweep1D` bookkeeping (loop names, counts, applying and undoing
  assignments) is what will justify it.
- More programs: qubit spectroscopy, Rabi, T1, T2.
- Markers to the NI DAQ through the PMOD pins (`Trigger(..., pins=[0])`), for
  correlating the NTD against qubit readout. Note the PMOD is **1.8 V
  LVCMOS18**, below the DAQ's TTL threshold, so it needs a level shifter —
  ideally an isolated one, so the marker cable does not add a ground loop.
