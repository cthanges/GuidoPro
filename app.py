import streamlit as st
import time
import pandas as pd
from src import data_loader
from src.simulator import SimpleSimulator
from src.analytics.pit_strategy import recommend_pit
from src.analytics.caution_handler import recommend_under_caution


st.set_page_config(page_title="Race Engineer MVP", layout='wide')

st.title('Race Engineer — Streamlit MVP')

with st.sidebar:
    st.header('Dataset')
    files = data_loader.list_lap_time_files()
    if files:
        choice = st.selectbox('Choose lap_time file', options=files)
    else:
        st.info('No lap_time files auto-detected in Datasets/. You can upload a CSV.')
        choice = st.file_uploader('Upload lap_time CSV', type=['csv'])

    speed = st.slider('Replay speed (laps/sec)', 0.1, 5.0, 1.0)
    target_stint = st.number_input('Target stint (laps)', min_value=1, max_value=100, value=20)
    pit_cost = st.number_input('Pit time cost (s)', min_value=1.0, max_value=120.0, value=20.0)
    degradation_rate = st.number_input('Tyre degradation (s/lap)', min_value=0.0, max_value=1.0, value=0.15, step=0.05)
    total_race_laps = st.number_input('Total race laps (optional)', min_value=0, max_value=200, value=50)

if not choice:
    st.warning('Please select or upload a lap_time CSV to begin.')
    st.stop()

if isinstance(choice, str):
    df = data_loader.load_lap_time(choice)
else:
    df = pd.read_csv(choice)

vehicle_ids = data_loader.vehicle_ids_from_lap_time(df)
vehicle = None
if vehicle_ids:
    vehicle = st.selectbox('Vehicle', vehicle_ids)
else:
    st.warning('No vehicle_id column found in the selected file; the app will use entire file rows.')

if vehicle:
    vdf = data_loader.filter_vehicle_laps(df, vehicle)
else:
    vdf = df

st.sidebar.markdown('---')
st.sidebar.write('Rows for selected vehicle: %d' % len(vdf))

sim = SimpleSimulator(vdf, speed=speed)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader('Replay')
    placeholder = st.empty()
    step_button = st.button('Step one lap')
    run_button = st.button('Run replay')

with col2:
    st.subheader('Pit Strategy')
    info_box = st.empty()

last_pit_lap = 0
last_laps = []
current_lap = 0

if 'sim_pos' not in st.session_state:
    st.session_state['sim_pos'] = 0

def render_row(row):
    # display some useful fields
    timestamp = row.get('timestamp') if 'timestamp' in row.index else None
    lap = int(row['lap']) if 'lap' in row.index and str(row['lap']).isdigit() else st.session_state['sim_pos']
    val = row.get('value') if 'value' in row.index else None
    return lap, timestamp, val


if step_button or run_button:
    try:
        if run_button:
            # run until end
            for row in sim.replay(delay_callback=time.sleep):
                lap, ts, val = render_row(row)
                st.session_state['sim_pos'] += 1
                # update last laps
                try:
                    last_laps.append(float(row.get('value', 0)))
                except Exception:
                    pass
                # limit history
                last_laps = last_laps[-5:]
                # compute recommendation with remaining laps
                remaining = total_race_laps - lap if total_race_laps > lap else None
                rec = recommend_pit(
                    lap, last_pit_lap, last_laps, 
                    target_stint=target_stint, 
                    pit_time_cost=pit_cost,
                    remaining_laps=remaining,
                    degradation_per_lap=degradation_rate
                )
                placeholder.metric('Lap', lap, delta=None)
                info_box.json(rec)
                # Attempt a safe rerun. Newer/older Streamlit builds sometimes
                # don't expose `st.experimental_rerun`. Use it when present,
                # otherwise raise the internal RerunException as a fallback.
                try:
                    # preferred public API (may not exist on some builds)
                    st.experimental_rerun()
                except Exception:
                    try:
                        # internal API fallback (works in many Streamlit versions)
                        from streamlit.runtime.scriptrunner.script_runner import RerunException

                        raise RerunException()
                    except Exception:
                        # Last-resort: set a flag so the UI can react without hard rerun.
                        st.session_state['_rerun_requested'] = True
        else:
            row = sim.next()
            lap, ts, val = render_row(row)
            st.session_state['sim_pos'] += 1
            try:
                last_laps.append(float(row.get('value', 0)))
            except Exception:
                pass
            last_laps = last_laps[-5:]
            remaining = total_race_laps - lap if total_race_laps > lap else None
            rec = recommend_pit(
                lap, last_pit_lap, last_laps, 
                target_stint=target_stint, 
                pit_time_cost=pit_cost,
                remaining_laps=remaining,
                degradation_per_lap=degradation_rate
            )
            placeholder.metric('Lap', lap, delta=None)
            info_box.json(rec)
    except StopIteration:
        st.info('Replay finished')

st.markdown('---')
st.subheader('Caution handling')
if st.button('Simulate Caution Now'):
    # get last recommendation from info_box by recomputing from state
    pos = st.session_state.get('sim_pos', 0)
    if pos > 0 and pos <= len(vdf):
        # recompute using last available lap
        row = vdf.iloc[min(pos - 1, len(vdf)-1)]
        lap = int(row['lap']) if 'lap' in row.index and str(row['lap']).isdigit() else pos
        remaining = total_race_laps - lap if total_race_laps > lap else None
        rec = recommend_pit(
            lap, last_pit_lap, last_laps, 
            target_stint=target_stint, 
            pit_time_cost=pit_cost,
            remaining_laps=remaining,
            degradation_per_lap=degradation_rate
        )
        caution = recommend_under_caution(rec, pit_time_cost=pit_cost)
        st.json({'recommendation': rec, 'caution_decision': caution})
    else:
        st.warning('No lap in flight to base a caution decision on')
