# rfsoc4x2

A QM-style configuration layer over [qickodes](https://github.com/aalto-qcd/qickodes)
for the RFSoC 4x2.

Describe the setup as **elements** (a qubit, a resonator) that own named
**pulses**, then write measurements in terms of those names instead of DAC and
ADC numbers. The layer also does the bookkeeping that is easy to forget:
the ADC frequency follows the resonator frequency, the DAC/ADC pair is matched,
and a qubit never gets an ADC.

## Requirements

- tProc **v2** firmware on the board (`v2r27` is what this was written against).
  The v2 API refuses to connect to v1 firmware.
- Python **3.10+** (qickodes itself needs 3.9+, but this package uses
  `X | None` annotations).

## Install on the lab computer

```bash
git clone <your-repo> PhD
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

## First connection

```python
from qcodes import Station
from rfsoc4x2 import RFSoC, RunConfig, Config, ResonatorSpec, ConstantPulseSpec
from rfsoc4x2.sequences import readout

config = Config(
    elements=[
        ResonatorSpec(name="r0", dac=0, adc=0, frequency=1e9,
                      readout_length=10e-6, nqz=1,
                      time_of_flight=0.0),      # measured below
    ],
    pulses=[
        ConstantPulseSpec(name="readout", element="r0", length=10e-6, gain=0.5),
    ],
)

rf = RFSoC("qi", "ip.of.board", Station(), "data/rfsoc.db", config, ns_port=8000)
print(rf.qi.soccfg)     # tProc type/revision, generators, readouts, output pins
```

`print(rf.qi.soccfg)` is the one command worth running first: it confirms the
tProc is `qick_processor` revision 27, and gives the DAC sample rates (which
decide the Nyquist zone), the generator types, and the marker pin list.

## Loopback test

Wire **DAC A → 20–30 dB attenuator → ADC A**. Do not connect them directly;
the ADC will clip.

Start at **500 MHz – 1 GHz**, not 6 GHz: it is safely in Nyquist zone 1 and well
inside the balun's 10 MHz – 10 GHz band, so one variable less.

**Step 1 — see the waveform.**

```python
rc = RunConfig(measurement_name="loopback", experiment_name="commissioning",
               sample_name="none", acquisition_mode="decimated",
               n_shots=1, soft_avgs=1)
rf.run(readout(rf.element("r0")), rc)
```

Look for a clean burst, not a clipped one. Read off **when the pulse arrives**
and put that number into `time_of_flight`.

**Step 2 — one IQ point.**

```python
rc = RunConfig(measurement_name="loopback_iq", experiment_name="commissioning",
               sample_name="none")          # accumulated, 1000 shots
rf.run(readout(rf.element("r0")), rc)
```

A stable complex number means the whole chain works: connection, firmware,
config, builders, elements, sequences and the database.

## Layout

| File | Holds |
|---|---|
| `specs.py` | dataclasses only. No qickodes import, no board needed |
| `config.py` | `Config` + validation (unique names, pulses point at real elements) |
| `build.py` | specs → live qickodes objects, one `singledispatch` builder per type |
| `elements.py` | the live `Element`: its DAC, ADC, frequency and pulses |
| `sequences.py` | `readout()`, `play()`, `wait()` → lists of macros |
| `rfsoc.py` | `RFSoC` (connection, elements, `run()`) and `RunConfig` |
| `sweeps.py` | **empty** — the sweep layer |
| `programs.py` | **empty** — punch-out, Rabi, T1 |

### Design rules

- **Specs are data.** `specs.py` never imports qickodes, so a config can be
  written, checked and version-controlled without a board.
- **`RFSoC` decides what to build and when; `build.py` knows how.** Adding a
  pulse type means one registered builder, nothing else changes.
- **Don't wrap what you don't need to change.** `element.pulse("readout")` *is*
  the qickodes `DacPulse`, so everything in the qickodes docs still applies,
  and `element.qi` reaches the `QickInstrument` through `dac.parent`.

## Gotchas worth remembering

- **`singledispatch` resolves every annotation** in a registered function, not
  just the first argument. Imports used in type hints must be real runtime
  imports, not under `if TYPE_CHECKING`. See the note at the top of `build.py`.
- **Pulse length vs envelope length.** `ConstantPulse.length` is the whole
  pulse; `FlatTopPulse.length` is only the flat part; `ArbitraryPulse` has no
  length at all — its duration is the envelope's.
- **Envelope parameters cannot be hardware-swept** (`sigma`, `delta`, `alpha`
  are `ManualParameter`). Pulse parameters can (`freq`, `phase`, `gain`,
  `length` are `SweepableParameter`).
- **Sweeping a readout frequency must move the ADC too.** This is why the
  resonator owns the frequency.
- **`final_delay` should be at least 5×T1**, ideally 10×. Too short biases a
  fitted T1 low, with no warning.

## Still to do

- `sweeps.py`: a `Sweep` spec plus a context manager that restores swept
  parameters afterwards. Without it, a Rabi leaves `QickSweep1D` sitting in
  `pi_pulse.gain` and the next measurement inherits it.
- `programs.py`: `punchout`, `rabi`, `t1`, each returning macros + a `RunConfig`.
- Calibration save/load, so a day's tuning survives to the next day.
