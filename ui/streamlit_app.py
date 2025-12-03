"""
ATE Report Generator - Streamlit UI
Simple interface for generating Auto-Technical Expertise reports
"""

import streamlit as st
import requests
import json
from datetime import datetime

# Page config
st.set_page_config(
    page_title="ATE Report Generator",
    page_icon="🚗",
    layout="wide"
)

# API endpoint (gateway running on rig)
API_BASE = "http://192.168.1.32:80"

st.title("🚗 Автотехническа Експертиза")
st.caption("AI-powered accident reconstruction report generator")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", value=API_BASE)

    st.markdown("---")
    st.markdown("### Collision Types")
    st.markdown("""
    - `left_turn` - Завой наляво
    - `right_turn` - Завой надясно
    - `rear_end` - Удар отзад
    - `head_on` - Челен удар
    - `side_impact` - Страничен удар
    """)

# Main content
tab1, tab2, tab3 = st.tabs(["📄 New Report", "📂 Load Result", "ℹ️ About"])

with tab1:
    st.header("Generate New ATE Report")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload PTP Protocol (text file)",
        type=['txt', 'json'],
        help="Upload the accident protocol document"
    )

    # Or paste text directly
    raw_text = st.text_area(
        "Or paste protocol text directly:",
        height=200,
        placeholder="ПРОТОКОЛ ЗА ПТП\n\nДата: ...\nМясто: ...\n\nУЧАСТНИЦИ:\nМПС А - ...\nМПС Б - ..."
    )

    # Extract from file if uploaded
    if uploaded_file:
        content = uploaded_file.read().decode('utf-8')
        if uploaded_file.name.endswith('.json'):
            try:
                data = json.loads(content)
                raw_text = data.get('raw_text', content)
            except:
                raw_text = content
        else:
            raw_text = content
        st.text_area("Uploaded content:", raw_text, height=150, disabled=True)

    st.markdown("---")

    # Editable parameters
    st.subheader("Vehicle Parameters (optional override)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**МПС А (Vehicle A)**")
        mass_a = st.number_input("Mass (kg)", value=0, key="mass_a", help="0 = extract from text")
        sigma_a = st.number_input("Post-impact travel σ (m)", value=0.0, key="sigma_a")
        skid_a = st.number_input("Skid distance (m)", value=0.0, key="skid_a")

    with col2:
        st.markdown("**МПС Б (Vehicle B)**")
        mass_b = st.number_input("Mass (kg)", value=0, key="mass_b", help="0 = extract from text")
        sigma_b = st.number_input("Post-impact travel σ (m)", value=0.0, key="sigma_b")
        skid_b = st.number_input("Skid distance (m)", value=0.0, key="skid_b")

    col3, col4 = st.columns(2)
    with col3:
        friction = st.number_input("Friction coefficient μ", value=0.7, min_value=0.1, max_value=1.0)
    with col4:
        collision_type = st.selectbox(
            "Collision type",
            ["auto-detect", "left_turn", "right_turn", "rear_end", "head_on", "side_impact", "perpendicular"],
            help="auto-detect extracts from Bulgarian text"
        )

    st.markdown("---")

    # Generate button
    if st.button("🚀 Generate ATE Report", type="primary", use_container_width=True):
        if not raw_text.strip():
            st.error("Please upload a file or paste protocol text")
        else:
            # Build request payload
            payload = {"raw_text": raw_text}

            # Add overrides if specified
            if mass_a > 0 or mass_b > 0:
                payload["vehicle_overrides"] = {}
                if mass_a > 0:
                    payload["vehicle_overrides"]["vehicle_a"] = {"mass_kg": mass_a}
                if mass_b > 0:
                    payload["vehicle_overrides"]["vehicle_b"] = {"mass_kg": mass_b}

            with st.spinner("Generating report... This may take 15-30 minutes"):
                try:
                    response = requests.post(
                        f"{api_url}/generate-ate-report",
                        json=payload,
                        timeout=3600  # 1 hour timeout
                    )

                    if response.status_code == 200:
                        result = response.json()
                        st.session_state['last_result'] = result
                        st.success(f"Report generated in {result.get('processing_time_seconds', 0):.1f} seconds!")
                        st.rerun()
                    else:
                        st.error(f"API Error: {response.status_code} - {response.text}")
                except requests.exceptions.Timeout:
                    st.error("Request timed out. Try again or check server status.")
                except requests.exceptions.ConnectionError:
                    st.error(f"Cannot connect to {api_url}. Is the gateway running?")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

with tab2:
    st.header("Load Previous Result")

    # File upload for result.json
    result_file = st.file_uploader(
        "Upload result.json",
        type=['json'],
        key="result_upload"
    )

    if result_file:
        try:
            result = json.loads(result_file.read().decode('utf-8'))
            st.session_state['last_result'] = result
            st.success("Result loaded!")
        except Exception as e:
            st.error(f"Error loading file: {e}")

    # Or load from path
    result_path = st.text_input(
        "Or enter path to result.json:",
        placeholder="/home/ubuntu/legal-ai/result.json"
    )

    if st.button("Load from path"):
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
                st.session_state['last_result'] = result
                st.success("Result loaded!")
        except FileNotFoundError:
            st.error("File not found")
        except Exception as e:
            st.error(f"Error: {e}")

with tab3:
    st.header("About")
    st.markdown("""
    ### ATE Report Generator

    This tool generates **Auto-Technical Expertise** (Автотехническа Експертиза) reports
    for traffic accident reconstruction using the **Momentum 360** physics method.

    #### Features:
    - Extract vehicle data from Bulgarian PTP protocols
    - Calculate pre-impact speeds using momentum conservation
    - Generate professional reports compliant with Наредба 24
    - Support for various collision types (left turn, rear-end, head-on, etc.)

    #### Physics Calculations:
    - Post-impact speed: `u = √(2μgσ)`
    - Momentum conservation in X and Y axes
    - Delta-V analysis
    - Stopping distance and reaction time

    #### API Endpoint:
    The backend runs on the GPU server at `192.168.1.32:80`
    """)

# Display result if available
if 'last_result' in st.session_state:
    st.markdown("---")
    st.header("📋 Report Result")

    result = st.session_state['last_result']

    # Metrics row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Processing Time", f"{result.get('processing_time_seconds', 0):.1f}s")
    with col2:
        st.metric("Sources Cited", len(result.get('sources_cited', [])))
    with col3:
        report_len = len(result.get('report_text', ''))
        st.metric("Report Length", f"{report_len:,} chars")

    # Report tabs
    report_tab1, report_tab2, report_tab3 = st.tabs(["📄 Rendered", "📝 Raw Markdown", "📚 Sources"])

    with report_tab1:
        st.markdown(result.get('report_text', 'No report text'))

    with report_tab2:
        st.code(result.get('report_text', ''), language='markdown')

    with report_tab3:
        sources = result.get('sources_cited', [])
        if sources:
            for src in sources:
                with st.expander(f"📖 {src.get('reference', 'Unknown')} (score: {src.get('score', 0):.2f})"):
                    st.json(src)
        else:
            st.info("No sources cited")

    # Download buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            "📥 Download Markdown",
            result.get('report_text', ''),
            file_name=f"ate_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown"
        )

    with col2:
        st.download_button(
            "📥 Download JSON",
            json.dumps(result, ensure_ascii=False, indent=2),
            file_name=f"ate_result_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json"
        )

    with col3:
        if st.button("🗑️ Clear Result"):
            del st.session_state['last_result']
            st.rerun()
