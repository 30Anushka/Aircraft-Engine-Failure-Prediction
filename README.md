# Aircraft Engine Remaining Useful Life (RUL) Prediction Dashboard

This project predicts the Remaining Useful Life (RUL) of aircraft engines using NASA’s C-MAPSS dataset and visualizes results in a real-time dashboard.

## Features
- Machine Learning model: Random Forest Regressor
- Real-time simulation of engine degradation
- Maintenance alerts based on predicted RUL
- Plotly and Matplotlib visualizations (auto fallback)
- Live deployment on Streamlit Cloud

## Run Locally
```bash
git clone https://github.com/30Anushka/aircraft-engine-failure-prediction
cd aircraft-engine-failure-prediction
pip install -r requirements.txt
streamlit run dashboard.py
