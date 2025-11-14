# Telemetry Data Guide

## Overview
GuidoAI includes comprehensive support for high-frequency telemetry data from race vehicles, with robust handling of known data quality issues.

## Data Files

### File Types
- **Lap Time Files**: `*lap_time*.csv` - Aggregated lap timing data
- **Lap Start/End Files**: `*lap_start*.csv`, `*lap_end*.csv` - Lap boundary events
- **Telemetry Files**: `*telemetry*.csv` - High-frequency sensor data (100+ Hz)

### Telemetry File Format
```csv
meta_time,timestamp,vehicle_id,lap,telemetry_name,telemetry_value,...
2025-09-06T18:40:41.926Z,2025-09-05T00:28:20.593Z,GR86-004-78,2,Speed,150
2025-09-06T18:40:41.926Z,2025-09-05T00:28:20.593Z,GR86-004-78,2,aps,85
```

**Key Columns:**
- `meta_time`: When message was received by system (reliable for ordering)
- `timestamp`: ECU timestamp from vehicle (may be inaccurate)
- `vehicle_id`: Format `GR86-{chassis}-{car_number}` (e.g., `GR86-004-78`)
- `lap`: Lap number (may be corrupted, see Known Issues)
- `telemetry_name`: Parameter name (e.g., `Speed`, `aps`, `accx_can`)
- `telemetry_value`: Numeric value

## Vehicle Identification

### Format: `GR86-{chassis}-{car_number}`
- **Chassis number**: Permanent identifier (e.g., `004`)
- **Car number**: Sticker on car (e.g., `78`)

### Special Cases
- `GR86-004-000`: Car number not yet assigned to ECU
  - Still uniquely identifiable by chassis `004`
  - May be updated to actual car number in later races

### Usage
```python
from src.telemetry_loader import parse_vehicle_id

vid = parse_vehicle_id("GR86-004-78")
print(vid.chassis_number)  # "004"
print(vid.car_number)      # "78"
print(vid.is_car_number_assigned)  # True

vid2 = parse_vehicle_id("GR86-002-000")
print(vid2.unique_id)  # "chassis-002" (fallback when car # unassigned)
```

## Known Data Quality Issues

### 1. Lap Counter Errors (Lap #32768)
**Issue:** ECU sometimes reports lap number as `32768` when counter is lost or corrupted.

**Handling:**
- Automatically detected by `is_valid_lap(lap)`
- Inferred from timestamps using `clean_lap_numbers(df)`
- Uses time gaps (>60s) to detect lap boundaries

**Example:**
```python
from src.telemetry_loader import load_telemetry

# Automatically cleans lap #32768 errors
df = load_telemetry("path/to/telemetry.csv", clean_data=True)
```

### 2. Timestamp Accuracy
**Issue:** ECU `timestamp` may not be accurate (clock drift, initialization errors).

**Best Practice:** Use `meta_time` for ordering events when available:
```python
df = df.sort_values('meta_time')  # Preferred
```

### 3. Missing or Partial Data
**Issue:** Not all parameters available for all vehicles/laps.

**Handling:**
```python
from src.telemetry_loader import validate_telemetry_quality

report = validate_telemetry_quality(df)
print(f"Available parameters: {report['available_parameters']}")
print(f"Warnings: {report['warnings']}")
```

## Telemetry Parameters

### Speed & Drivetrain
| Parameter | Unit | Description |
|-----------|------|-------------|
| `Speed` | km/h | Actual vehicle speed |
| `Gear` | gear | Current gear selection |
| `nmot` | rpm | Engine RPM |

### Throttle & Braking
| Parameter | Unit | Description |
|-----------|------|-------------|
| `ath` | % | Throttle blade position (0=closed, 100=wide open) |
| `aps` | % | Accelerator pedal position (0=none, 100=full) |
| `pbrake_f` | bar | Front brake pressure |
| `pbrake_r` | bar | Rear brake pressure |

### Acceleration & Steering
| Parameter | Unit | Description |
|-----------|------|-------------|
| `accx_can` | G | Forward/backward acceleration (+forward, -braking) |
| `accy_can` | G | Lateral acceleration (+left turn, -right turn) |
| `Steering_Angle` | degrees | Steering wheel angle (0=straight, +CW, -CCW) |

### Position & Lap Data
| Parameter | Unit | Description |
|-----------|------|-------------|
| `VBOX_Long_Minutes` | degrees | GPS longitude |
| `VBOX_Lat_Min` | degrees | GPS latitude |
| `Laptrigger_lapdist_dls` | meters | Distance from start/finish line |

## Usage Examples

### Load and Clean Telemetry
```python
from src.telemetry_loader import load_telemetry, get_vehicle_ids

# Load with automatic cleaning
df = load_telemetry("Datasets/barber/.../telemetry.csv", clean_data=True)

# Get all vehicles
vehicles = get_vehicle_ids(df)
for v in vehicles:
    print(f"{v.chassis_number}: {v}")
```

### Filter by Vehicle and Parameter
```python
from src.telemetry_loader import get_vehicle_telemetry

# Get all Speed data for specific vehicle in laps 5-10
speed_data = get_vehicle_telemetry(
    df, 
    vehicle_id="GR86-004-78",
    parameter="Speed",
    lap_range=(5, 10)
)

print(speed_data[['meta_time', 'lap', 'telemetry_value']].head())
```

### Convert to Wide Format
```python
from src.telemetry_loader import telemetry_to_wide_format

# Convert from long format (one row per parameter)
# to wide format (one row per timestamp with all parameters as columns)
wide_df = telemetry_to_wide_format(df)

# Now you can access parameters directly
print(wide_df[['meta_time', 'Speed', 'aps', 'nmot']].head())
```

### Data Quality Check
```python
from src.telemetry_loader import validate_telemetry_quality

report = validate_telemetry_quality(df)
print(f"Total rows: {report['total_rows']}")
print(f"Invalid laps: {report['invalid_laps']}")
print(f"Available parameters: {len(report['available_parameters'])}")

if report['warnings']:
    for warning in report['warnings']:
        print(f"⚠ {warning}")
```

## Integration with Existing Code

### Lap-Level Analysis (Current)
The existing `src.data_loader` handles lap-aggregated data:
```python
from src.data_loader import load_lap_time

lap_df = load_lap_time("path/to/lap_time.csv")
```

### Telemetry-Level Analysis (New)
Use `src.telemetry_loader` for high-frequency data:
```python
from src.telemetry_loader import load_telemetry

telem_df = load_telemetry("path/to/telemetry.csv")
```

### Combined Workflow
```python
# 1. Load lap times for high-level strategy
from src.data_loader import load_lap_time
laps = load_lap_time("lap_time.csv")

# 2. Load telemetry for detailed analysis
from src.telemetry_loader import load_telemetry, get_vehicle_telemetry
telemetry = load_telemetry("telemetry.csv")

# 3. Analyze specific vehicle in target lap
vehicle_telem = get_vehicle_telemetry(
    telemetry, 
    "GR86-004-78", 
    lap_range=(15, 15)  # Focus on lap 15
)

# 4. Check tire degradation from lateral accel
accy = get_vehicle_telemetry(vehicle_telem, "GR86-004-78", parameter="accy_can")
print(f"Max lateral G in lap 15: {accy['telemetry_value'].max()}")
```

## Future Enhancements

### Telemetry-Based Degradation Model
Instead of fixed `degradation_per_lap`, use actual telemetry:
- Track lateral/longitudinal G-forces per lap
- Estimate tire wear from peak accel events
- Model temperature effects from brake pressure

### Real-Time Anomaly Detection
- Monitor sudden drop in `nmot` (mechanical issue)
- Detect lockups from `pbrake_f/r` spikes
- Flag track limits violations from GPS + lap distance

### Sector-Level Optimization
- Split lap into sectors using `Laptrigger_lapdist_dls`
- Compute sector times and compare with baseline
- Optimize pit timing based on sector degradation patterns

## Testing
Run comprehensive tests:
```powershell
pytest tests/test_telemetry_loader.py -v
```

Test with real data:
```powershell
python test_telemetry_real_data.py
```
