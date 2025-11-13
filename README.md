# Race Engineer — Streamlit MVP

Real-time analytics tool that simulates race-engineering decision-making for optimal pit strategy and caution responses.

## Features

- **Optimal Pit Window Calculator**: Multi-lap optimizer that evaluates candidate pit stops and selects the timing that minimizes total race time
- **Tyre Degradation Model**: Linear degradation model (configurable seconds/lap) to estimate lap time increases
- **Caution Handler**: Adjusts pit recommendations when yellow flags reduce pit-time cost
- **Lap-by-Lap Replay**: Step through or auto-replay race data with real-time recommendations
- **Interactive Tuning**: Adjust pit cost, degradation rate, target stint, and race length via UI controls

## How It Works

The optimizer evaluates multiple candidate pit laps (e.g., lap 11, 12, 13...) and for each strategy computes:
1. Time remaining on worn tyres before pit
2. Pit time cost
3. Time on fresh tyres after pit (with degradation reset)
4. Total expected race time

It selects the pit lap that minimizes total time and reports the expected time saved vs. no-pit baseline.

## How to Run (Local)

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the Streamlit app:

```powershell
streamlit run app.py
```

3. Run tests:

```powershell
pip install pytest
pytest -v
```

## Usage Notes
- The app will attempt to find lap_time CSV files under `Datasets/` automatically.
- Select a file and vehicle ID, then step through laps or run a replay.
- Use the sidebar to tune:
  - **Target stint**: Fallback max stint length (laps)
  - **Pit time cost**: Time lost in pit (seconds)
  - **Tyre degradation**: Lap time increase per lap of tyre age (seconds/lap)
  - **Total race laps**: Used to compute remaining laps for optimization window

## Files

- `src/data_loader.py` — Helpers to find/load lap_time CSVs
- `src/telemetry_loader.py` — **Robust telemetry data loader** with data quality handling (see [TELEMETRY_GUIDE.md](TELEMETRY_GUIDE.md))
- `src/simulator.py` — Lap-level replay engine
- `src/analytics/pit_strategy.py` — **Multi-lap pit window optimizer** with degradation model
- `src/analytics/caution_handler.py` — Caution decision logic
- `tests/test_pit_strategy.py` — Unit tests for pit optimizer
- `tests/test_telemetry_loader.py` — Unit tests for telemetry loader (14 tests)

## Telemetry Support

The project includes comprehensive telemetry data handling with robust data quality fixes:
- **Vehicle ID parsing**: Handles `GR86-{chassis}-{car_number}` format
- **Lap #32768 detection**: Automatically detects and corrects ECU lap counter errors
- **Timestamp handling**: Distinguishes `meta_time` (reliable) from `timestamp` (ECU)
- **14 telemetry parameters**: Speed, throttle, braking, acceleration, steering, GPS, etc.
- **Data validation**: Quality reports with warnings and diagnostics

See [TELEMETRY_GUIDE.md](TELEMETRY_GUIDE.md) for detailed documentation and examples.

## Next Steps

- Integrate telemetry-based degradation (actual G-forces vs fixed rate)
- Add sector-level replay using distance-from-start telemetry
- Incorporate fuel model and compound-specific degradation curves
- Traffic and field-position modeling for undercut/overcut scenarios
- Probabilistic caution modeling (expected value with risk)
- Monte Carlo simulation for uncertainty quantification
- Richer UI with degradation charts and pit window visualization
