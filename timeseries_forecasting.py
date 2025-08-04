import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
from tensorflow.keras.models import load_model
from tcn import TCN
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Judul aplikasi
st.title("🔮 Prediksi Tag Value 10 Menit Ke Depan (Tanpa Noise)")

# Konstanta
WINDOW_SIZE = 60       # 10 menit terakhir (60 data point @10 detik)
FUTURE_STEPS = 60      # Prediksi 10 menit ke depan

# Load model dan scaler (dengan cache)
@st.cache_resource
def load_artifacts():
    try:
        model = load_model("my_model.keras", compile=False, custom_objects={"TCN": TCN})
        scaler = joblib.load("scalercp.joblib")
        return model, scaler
    except Exception as e:
        st.error(f"❌ Gagal memuat model atau scaler: {e}")
        st.stop()

model, scaler = load_artifacts()

# Upload file CSV
uploaded_file = st.file_uploader("📂 Upload File CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    try:
        # Konversi dan urutkan berdasarkan waktu
        df['ddate'] = pd.to_datetime(df['ddate'])
        df = df.sort_values('ddate').reset_index(drop=True)

        st.subheader("📊 Data Terakhir:")
        st.dataframe(df.tail(5))

        if len(df) < WINDOW_SIZE:
            st.error(f"❌ Data kurang. Minimal {WINDOW_SIZE} baris diperlukan.")
        else:
            # Ambil WINDOW_SIZE terakhir dan scaling
            last_values = df['tag_value'].values[-WINDOW_SIZE:]
            scaled_input = scaler.transform(last_values.reshape(-1, 1))

            # Prediksi 60 langkah ke depan
            forecast_scaled = []
            last_window = scaled_input.copy()

            for _ in range(FUTURE_STEPS):
                input_data = last_window.reshape((1, WINDOW_SIZE, 1))
                next_pred = model.predict(input_data, verbose=0)[0, 0]
                forecast_scaled.append(next_pred)
                last_window = np.append(last_window[1:], [[next_pred]], axis=0)

            # Invers skala ke nilai asli
            forecast_actual = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()

            # Buat timestamp prediksi
            last_time = df['ddate'].iloc[-1]
            future_times = [last_time + timedelta(seconds=10 * (i + 1)) for i in range(FUTURE_STEPS)]

            forecast_df = pd.DataFrame({
                "Tanggal": future_times,
                "Prediksi Tag Value (Tanpa Noise)": forecast_actual
            })

            # Tampilkan hasil
            st.subheader("📈 Grafik Prediksi (Tanpa Noise)")
            st.line_chart(forecast_df.set_index("Tanggal"))

            st.subheader("📋 Tabel Prediksi (Tanpa Noise)")
            st.dataframe(forecast_df)

            # Evaluasi Akurasi: gunakan 60 data aktual terakhir
            if len(df) >= WINDOW_SIZE + FUTURE_STEPS:
                actual_future = df['tag_value'].values[-FUTURE_STEPS:]
                mae = mean_absolute_error(actual_future, forecast_actual)
                rmse = np.sqrt(mean_squared_error(actual_future, forecast_actual))

                st.subheader("📉 Evaluasi Model (Data Uji)")
                st.markdown(f"""
                - *MAE (Mean Absolute Error)*: `{mae:.4f}`
                - *RMSE (Root Mean Squared Error)*: `{rmse:.4f}`
                """)
            else:
                st.warning("⚠️ Tidak cukup data untuk evaluasi akurasi (diperlukan minimal 60 baris aktual setelah input).")

    except Exception as e:
        st.error(f"❌ Error saat memproses data: {e}")
