# Project: Campus Entity Resolution & Security Monitoring System (Streamlit)
# Folder structure and files are included below. Copy each section into files with the paths shown.

# -----------------------------
# FILE: README.md
# -----------------------------
"""
# Campus Entity Resolution & Security Monitoring System (Streamlit)

## What's included
- `app.py` — Streamlit app (entrypoint)
- `src/entity_resolution.py` — simple entity resolution utilities
- `src/timeline_generator.py` — build timelines from logs
- `src/predictor.py` — simple predictive monitoring & alerts
- `data/sample_students.csv` — sample profiles
- `data/sample_swipes.csv` — sample swipe logs
- `data/sample_wifi.csv` — sample wifi logs
- `requirements.txt` — python packages
- `report_template.md` — technical report template

## Run locally
1. Create a venv and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. Run the Streamlit app:

```bash
streamlit run app.py
```

3. Open the local URL printed by Streamlit (usually http://localhost:8501).

## Demo video
Record 3–5 minute video showing: dataset upload, searching an entity, timeline, prediction, and alert.

"""

# -----------------------------
# FILE: requirements.txt
# -----------------------------
"""
streamlit
pandas
numpy
scikit-learn
python-dateutil
rapidfuzz
"""

# -----------------------------
# FILE: data/sample_students.csv
# -----------------------------
"""
student_id,name,email,card_id,department
S001,Ankit Sharma,ankit.sharma@campus.edu,CARD100,CS
S002,Ankita Sharma,ankita.sharma@campus.edu,CARD101,EE
S003,Dr. Ramesh Kumar,ramesh.kumar@campus.edu,CARD200,Admin
"""

# -----------------------------
# FILE: data/sample_swipes.csv
# -----------------------------
"""
card_id,location_id,timestamp
CARD100,Gate-A,2025-10-06T07:45:00
CARD101,Library,2025-10-06T09:30:00
CARD100,Lab-3,2025-10-06T10:50:00
CARD200,Admin-Block,2025-10-05T18:10:00
"""

# -----------------------------
# FILE: data/sample_wifi.csv
# -----------------------------
"""
device_hash,ap_id,timestamp
dev_abc123,AP-1,2025-10-06T08:00:00
dev_def456,AP-2,2025-10-06T11:00:00
dev_abc123,AP-3,2025-10-06T12:00:00
"""

# -----------------------------
# FILE: src/entity_resolution.py
# -----------------------------
"""
Simple entity resolution utilities. Combines exact ID matches and fuzzy name matching.
"""
from rapidfuzz import fuzz
import pandas as pd

def load_profiles(path):
    return pd.read_csv(path)

def resolve_by_card(card_id, students_df):
    matches = students_df[students_df['card_id'] == card_id]
    if not matches.empty:
        return matches.iloc[0].to_dict()
    return None

def fuzzy_name_search(name, students_df, threshold=85):
    best = None
    best_score = 0
    for _, row in students_df.iterrows():
        score = fuzz.token_sort_ratio(name.lower(), str(row['name']).lower())
        if score > best_score:
            best_score = score
            best = row
    if best_score >= threshold:
        out = best.to_dict()
        out['match_score'] = best_score
        return out
    return None

def resolve_entity(record, students_df):
    # Try direct id
    if 'card_id' in record and pd.notna(record['card_id']):
        r = resolve_by_card(record['card_id'], students_df)
        if r:
            r['resolve_method'] = 'card_id'
            r['confidence'] = 1.0
            return r
    # Try email
    if 'email' in record and pd.notna(record['email']):
        matches = students_df[students_df['email'] == record['email']]
        if not matches.empty:
            r = matches.iloc[0].to_dict()
            r['resolve_method'] = 'email'
            r['confidence'] = 1.0
            return r
    # Try fuzzy name
    if 'name' in record and pd.notna(record['name']):
        r = fuzzy_name_search(record['name'], students_df)
        if r:
            r['resolve_method'] = 'fuzzy_name'
            r['confidence'] = r.get('match_score', 90)/100.0
            return r
    # fallback
    return {'name': record.get('name', 'Unknown'), 'resolve_method': 'unknown', 'confidence': 0.0}


# -----------------------------
# FILE: src/timeline_generator.py
# -----------------------------
"""
Builds a chronological timeline for an entity from multiple log sources.
"""
import pandas as pd
from dateutil import parser


def parse_timestamp(ts):
    try:
        return parser.isoparse(ts)
    except Exception:
        return pd.NaT


def build_timeline(entity, swipes_df=None, wifi_df=None):
    events = []
    # Swipes
    if swipes_df is not None and 'card_id' in entity:
        rows = swipes_df[swipes_df['card_id'] == entity['card_id']]
        for _, r in rows.iterrows():
            events.append({'time': parse_timestamp(r['timestamp']), 'type': 'swipe', 'detail': r['location_id']})
    # WiFi (we don't have device mapping here — include as general nearby events)
    if wifi_df is not None:
        for _, r in wifi_df.iterrows():
            events.append({'time': parse_timestamp(r['timestamp']), 'type': 'wifi', 'detail': r['ap_id']})
    # Sort and return
    df = pd.DataFrame(events)
    if df.empty:
        return df
    df = df.dropna(subset=['time']).sort_values('time')
    df['time_str'] = df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    return df


# -----------------------------
# FILE: src/predictor.py
# -----------------------------
"""
Simple predictor: when data is missing, use last-seen location or AP as prediction.
Also detect inactivity > 12 hours.
"""
from datetime import datetime, timedelta


def predict_last_known(timeline_df, current_time=None):
    if current_time is None:
        current_time = datetime.utcnow()
    if timeline_df is None or timeline_df.empty:
        return {'prediction': None, 'confidence': 0.0, 'reason': 'no data'}
    last = timeline_df.iloc[-1]
    last_time = last['time'].to_pydatetime()
    # confidence decays with time
    hours_passed = (current_time - last_time).total_seconds()/3600.0
    confidence = max(0.1, 1.0 - 0.05 * hours_passed)
    return {'prediction': last['detail'], 'confidence': round(confidence, 2), 'last_seen': last_time.isoformat(), 'hours_since': round(hours_passed,2)}


def check_inactivity(timeline_df, threshold_hours=12, current_time=None):
    if current_time is None:
        current_time = datetime.utcnow()
    if timeline_df is None or timeline_df.empty:
        return True, None
    last_time = timeline_df.iloc[-1]['time'].to_pydatetime()
    hours = (current_time - last_time).total_seconds()/3600.0
    return hours >= threshold_hours, hours


# -----------------------------
# FILE: app.py
# -----------------------------
"""
Streamlit app tying everything together.
"""
import streamlit as st
import pandas as pd
from src.entity_resolution import load_profiles, resolve_entity
from src.timeline_generator import build_timeline
from src.predictor import predict_last_known, check_inactivity
from datetime import datetime

st.set_page_config(page_title='Campus ER & Security Monitor', layout='wide')

st.title('Campus Entity Resolution & Security Monitoring')

# Load sample data
@st.cache_data
def load_data():
    students = pd.read_csv('data/sample_students.csv')
    swipes = pd.read_csv('data/sample_swipes.csv')
    wifi = pd.read_csv('data/sample_wifi.csv')
    return students, swipes, wifi

students, swipes, wifi = load_data()

# Sidebar input
st.sidebar.header('Search Entity')
search_name = st.sidebar.text_input('Name (optional)')
search_card = st.sidebar.text_input('Card ID (optional)')
search_email = st.sidebar.text_input('Email (optional)')

if st.sidebar.button('Resolve & Show Timeline'):
    query = {'name': search_name.strip() if search_name else None, 'card_id': search_card.strip() if search_card else None, 'email': search_email.strip() if search_email else None}
    entity = resolve_entity(query, students)
    st.subheader('Resolved Entity')
    st.write(entity)

    timeline = build_timeline(entity, swipes_df=swipes, wifi_df=wifi)
    if timeline.empty:
        st.info('No timeline events found for this entity.')
    else:
        st.subheader('Timeline')
        st.dataframe(timeline[['time_str', 'type', 'detail']].sort_values('time'))

    # Predictions & alerts
    pred = predict_last_known(timeline, current_time=datetime.utcnow())
    st.subheader('Prediction (if data missing)')
    st.write(pred)

    inactive, hours = check_inactivity(timeline, threshold_hours=12, current_time=datetime.utcnow())
    if inactive:
        st.error(f'ALERT: No observations in last 12 hours (last seen {hours:.1f} hours ago).')
    else:
        st.success(f'Entity observed {hours:.2f} hours ago — no alert.')

# Show sample tables
with st.expander('View sample datasets'):
    st.write('Students')
    st.dataframe(students)
    st.write('Swipes')
    st.dataframe(swipes)
    st.write('WiFi')
    st.dataframe(wifi)

# Quick demo buttons
st.markdown('---')
col1, col2 = st.columns(2)
if col1.button('Demo: Resolve Ankit by card'):
    st.experimental_rerun()

# Footer
st.markdown('---')
st.caption('Prototype for Round 1 submission — customize with the provided synthetic dataset.')

# -----------------------------
# FILE: report_template.md
# -----------------------------
"""
# Technical Report Template

## 1. Abstract
(3-4 lines summarizing the system and results)

## 2. Objectives
(What the project aims to solve)

## 3. System Architecture
(Overview diagram + components)

## 4. Data & Preprocessing
(Describe synthetic dataset, fields, cleaning)

## 5. Entity Resolution Approach
(Exact matches, fuzzy matching, confidence scoring)

## 6. Timeline Generation
(Method to combine logs, ordering, summarization)

## 7. Predictive Monitoring
(How predictions are made and explained)

## 8. Results
(Example queries, timelines, accuracy / qualitative analysis)

## 9. Privacy & Failure Modes
(Discuss privacy safeguards and what could go wrong)

## 10. Conclusion & Future Work

"""

# End of document
# You can copy the sections into files in a repo. Good luck with the hackathon!
