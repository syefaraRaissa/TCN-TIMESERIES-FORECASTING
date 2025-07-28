import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from datetime import timedelta

# Load model dan scaler
@st.cache_resource
def load_artifacts():
    model = load_model("tcn_timeseries_model.keras", compile=False)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

# Konstanta
WINDOW_SIZE = 60
FUTURE_STEPS = 60

st.title("🔮 Prediksi Tag Value 10 Menit ke Depan (per 10 Detik)")

uploaded_file = st.file_uploader("📂 Upload File CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    try:
        df['ddate'] = pd.to_datetime(df['ddate'])
        df = df.sort_values('ddate').reset_index(drop=True)
        st.subheader("Data Terbaru:")
        st.dataframe(df.tail(5))

        # Ambil window terakhir
        last_values = df['tag_value'].values[-WINDOW_SIZE:]
        if len(last_values) < WINDOW_SIZE:
            st.error(f"Data kurang. Butuh minimal {WINDOW_SIZE} baris data.")
        else:
            scaled_input = scaler.transform(last_values.reshape(-1, 1)).reshape(1, WINDOW_SIZE, 1)
            forecast = []
            current_input = scaled_input

            for _ in range(FUTURE_STEPS):
                pred = model.predict(current_input, verbose=0)[0, 0]
                forecast.append(pred)
                current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)

            forecast_actual = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
            last_time = df['ddate'].iloc[-1]
            future_times = [last_time + timedelta(seconds=10*(i+1)) for i in range(FUTURE_STEPS)]

            result_df = pd.DataFrame({
                'Datetime': future_times,
                'Prediksi Tag Value': forecast_actual.flatten()
            })

            st.subheader("📈 Hasil Prediksi")
            st.line_chart(result_df.set_index("Datetime"))
            st.dataframe(result_df)

    except Exception as e:
        st.error(f"Gagal membaca data. Pastikan format kolom: `ddate`, `tag_value`.\n\nError: {e}")
