import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
from tensorflow.keras.models import load_model
from tcn import TCN
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

st.title("🔮 Prediksi Tag Value 10 Menit Ke Depan (per 10 Detik)")

@st.cache_resource
def load_artifacts():
    try:
        model = load_model("/mnt/data/tcn_timeseries_model.keras", compile=False, custom_objects={"TCN": TCN})
        scaler = joblib.load("/mnt/data/scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"Gagal memuat model atau scaler: {e}")
        st.stop()

# Load model dan scaler
model, scaler = load_artifacts()

WINDOW_SIZE = 60     # 10 menit sebelumnya
FUTURE_STEPS = 60    # 10 menit ke depan
MIN_DATA = WINDOW_SIZE + FUTURE_STEPS  # minimal 120 data

uploaded_file = st.file_uploader("📂 Upload File CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    try:
        df['ddate'] = pd.to_datetime(df['ddate'])
        df = df.sort_values('ddate').reset_index(drop=True)

        st.subheader("📊 Data Terakhir:")
        st.dataframe(df.tail(5))

        if len(df) < MIN_DATA:
            st.error(f"❌ Minimal {MIN_DATA} baris data diperlukan untuk prediksi dan evaluasi.")
        else:
            # Ambil 60 data terakhir untuk input dan 60 setelahnya untuk evaluasi
            last_input_values = df['tag_value'].values[-MIN_DATA:-FUTURE_STEPS]
            actual_future_values = df['tag_value'].values[-FUTURE_STEPS:]

            scaled_input = scaler.transform(last_input_values.reshape(-1, 1)).reshape(1, WINDOW_SIZE, 1)

            forecast = []
            current_input = scaled_input.copy()

            for _ in range(FUTURE_STEPS):
                pred = model.predict(current_input, verbose=0)[0, 0]
                forecast.append(pred)
                current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)

            forecast_actual = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

            # Waktu prediksi
            last_time = df['ddate'].iloc[-FUTURE_STEPS - 1]
            future_times = [last_time + timedelta(seconds=10 * (i + 1)) for i in range(FUTURE_STEPS)]

            result_df = pd.DataFrame({
                'Datetime': future_times,
                'Prediksi Tag Value': forecast_actual.flatten(),
                'Aktual': actual_future_values
            })

            # Grafik 1: Prediksi saja
            st.subheader("📈 Grafik Prediksi 10 Menit Ke Depan")
            st.line_chart(result_df.set_index("Datetime")[["Prediksi Tag Value"]])

            # Grafik 2: Gabungan aktual vs prediksi
            st.subheader("📊 Grafik Gabungan Aktual vs Prediksi")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(result_df['Datetime'], result_df['Aktual'], label="Aktual", color="blue")
            ax.plot(result_df['Datetime'], result_df['Prediksi Tag Value'], label="Prediksi", color="red")
            ax.set_title("Perbandingan Prediksi dan Aktual")
            ax.set_xlabel("Waktu")
            ax.set_ylabel("Tag Value")
            ax.legend()
            st.pyplot(fig)

            # Evaluasi
            mae = mean_absolute_error(result_df['Aktual'], result_df['Prediksi Tag Value'])
            rmse = np.sqrt(mean_squared_error(result_df['Aktual'], result_df['Prediksi Tag Value']))

            st.subheader("📉 Evaluasi Model (Data Uji)")
            st.markdown(f"""
            - **MAE (Mean Absolute Error)**: `{mae:.4f}`
            - **RMSE (Root Mean Squared Error)**: `{rmse:.4f}`
            """)

            # Tabel Prediksi
            st.subheader("📋 Tabel Prediksi dan Aktual")
            st.dataframe(result_df)

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat memproses data: {e}")
