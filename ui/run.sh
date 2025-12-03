#!/bin/bash
# Run Streamlit ATE Report UI

cd "$(dirname "$0")"

# Install deps if needed
pip install -q -r requirements.txt

# Run streamlit
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
