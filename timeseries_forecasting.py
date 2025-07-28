import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from datetime import timedelta
import matplotlib.pyplot as plt

# Load model dan scaler
model = load_model('tcn_timeseries_model.keras', compile=False)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Konstanta
WINDOW_SIZE = 60  # disesuaikan
FUTURE_STEPS = 60  # 10 menit = 60 titik (per 10 detik)

st.title("Prediksi Tag Value (TCN Forecasting)")

uploaded_file = st.file_uploader("Upload file CSV berisi data terbaru", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['ddate'] = pd.to_datetime(df['ddate'])
    df = df.sort_values('ddate').reset_index(drop=True)

    st.write("📊 Data Terbaru:")
    st.dataframe(df.tail(5))

    # Ambil nilai tag_value terakhir sebanyak window_size
    last_values = df['tag_value'].values[-WINDOW_SIZE:]
    if len(last_values) < WINDOW_SIZE:
        st.error(f"Data kurang. Butuh minimal {WINDOW_SIZE} baris.")
    else:
        # Scaling
        scaled_input = scaler.transform(last_values.reshape(-1, 1)).reshape(1, WINDOW_SIZE, 1)

        forecast = []
        current_input = scaled_input

        for _ in range(FUTURE_STEPS):
            next_pred_scaled = model.predict(current_input, verbose=0)[0, 0]
            forecast.append(next_pred_scaled)
            current_input = np.append(current_input[:, 1:, :], [[[next_pred_scaled]]], axis=1)

        # Inverse transform
        forecast_actual = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

        # Buat datetime ke depan
        last_time = df['ddate'].iloc[-1]
        future_times = [last_time + timedelta(seconds=10 * (i + 1)) for i in range(FUTURE_STEPS)]

        # Tampilkan hasil
        result_df = pd.DataFrame({
            'Datetime': future_times,
            'Prediksi Tag Value': forecast_actual.flatten()
        })

        st.subheader("🔮 Hasil Prediksi 10 Menit Ke Depan")
        st.dataframe(result_df)

        # Visualisasi
        st.line_chart(result_df.set_index('Datetime'))

