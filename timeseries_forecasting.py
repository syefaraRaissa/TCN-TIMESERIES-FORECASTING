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

# Fungsi caching untuk load model dan scaler
@st.cache_resource
def load_artifacts():
    try:
        model = load_model("tcn_timeseries_model.keras", compile=False, custom_objects={"TCN": TCN})
        scaler = joblib.load("scaler.joblib")
        return model, scaler
    except Exception as e:
        st.error(f"Gagal memuat model atau scaler: {e}")
        st.stop()

# Muat model dan scaler
model, scaler = load_artifacts()

# Parameter
WINDOW_SIZE = 30      # jumlah data input (10 detik * 30 = 5 menit sebelumnya)
FUTURE_STEPS = 60     # jumlah langkah prediksi (10 detik * 60 = 10 menit ke depan)

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload File CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    try:
        # Preprocessing kolom waktu dan urutkan
        df['ddate'] = pd.to_datetime(df['ddate'])
        df = df.sort_values('ddate').reset_index(drop=True)

        st.subheader("📊 Data Terakhir:")
        st.dataframe(df.tail(5))

        if len(df) < WINDOW_SIZE:
            st.error(f"❌ Data kurang. Minimal {WINDOW_SIZE} baris diperlukan.")
        else:
            # Ambil WINDOW_SIZE nilai terakhir sebagai input
            last_values = df['tag_value'].values[-WINDOW_SIZE:]
            scaled_input = scaler.transform(last_values.reshape(-1, 1)).reshape(1, WINDOW_SIZE, 1)

            forecast = []
            current_input = scaled_input

            # Prediksi FUTURE_STEPS langkah ke depan
            for _ in range(FUTURE_STEPS):
                pred = model.predict(current_input, verbose=0)[0, 0]
                forecast.append(pred)
                # Sliding window
                current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)

            # Kembalikan ke skala asli
            forecast_actual = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

            last_time = df['ddate'].iloc[-1]
            future_times = [last_time + timedelta(seconds=10 * (i + 1)) for i in range(FUTURE_STEPS)]

            result_df = pd.DataFrame({
                'Datetime': future_times,
                'Prediksi Tag Value': forecast_actual.flatten()
            })

            # ================= GRAFIK MIRIP GAMBAR =================
            st.subheader("📈 Grafik Prediksi")
            hist_times = df['ddate'].iloc[-WINDOW_SIZE:]
            hist_values = df['tag_value'].iloc[-WINDOW_SIZE:]

            fig, ax = plt.subplots(figsize=(10, 5))

            # Plot data historis
            ax.plot(hist_times, hist_values, color="blue", label="Data Historis")

            # Plot prediksi
            ax.plot(result_df['Datetime'], result_df['Prediksi Tag Value'], color="red", label="Prediksi")

            # Style biar mirip contoh
            ax.set_title("📊 Grafik Prediksi", fontsize=14, fontweight="bold")
            ax.set_xlabel("Waktu")
            ax.set_ylabel("Tag Value")
            ax.grid(True, linestyle="--", alpha=0.7)

            # Legend di kanan atas dengan frame
            ax.legend(loc="upper right", fontsize=10, frameon=True)

            # Tambahkan border hitam (frame) di sekitar grafik
            for spine in ax.spines.values():
                spine.set_edgecolor("black")
                spine.set_linewidth(1.2)

            st.pyplot(fig)
            # ========================================================

            st.subheader("📋 Tabel Prediksi")
            st.dataframe(result_df)

            # Evaluasi jika data cukup
            if len(df) >= WINDOW_SIZE + FUTURE_STEPS:
                actual_future = df['tag_value'].values[-FUTURE_STEPS:]
                mae = mean_absolute_error(actual_future, forecast_actual)
                rmse = np.sqrt(mean_squared_error(actual_future, forecast_actual))

                st.subheader("📉 Evaluasi Model (Data Uji)")
                st.markdown(f"""
                - **MAE (Mean Absolute Error)**: {mae:.4f}
                - **RMSE (Root Mean Squared Error)**: {rmse:.4f}
                """)
            else:
                st.warning("⚠️ Tidak cukup data untuk evaluasi MAE dan RMSE (diperlukan minimal 60 baris data aktual setelah input).")

    except Exception as e:
        st.error(f"❌ Error saat memproses data: {e}")
