import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="SPK Rekomendasi Istirahat Screen Time", layout="wide")

st.title("Sistem Pendukung Keputusan Rekomendasi Istirahat Screen Time")
st.markdown("""
Aplikasi ini menggunakan metode K-Means Clustering untuk memberikan rekomendasi waktu istirahat
berdasarkan pola penggunaan smartphone Anda.  
Silakan isi data penggunaan harian pada kolom di bawah.
""")

# ---------------------------------------------------------
# 1. Load dataset dari file lokal
# ---------------------------------------------------------
DATA_PATH = "data/mobile_screen_time.csv"

try:
    df = pd.read_csv(DATA_PATH)
except:
    st.error(f"File dataset tidak ditemukan. Pastikan file berada di folder: {DATA_PATH}")
    st.stop()

# Kolom numerik yang dipakai dalam proses klustering
numeric_cols = [
    "Daily_Screen_Time_Hours",
    "Screen_Unlocks_Per_Day",
    "App_Notifications_Received"
]

df_clean = df[numeric_cols].dropna()

# ---------------------------------------------------------
# 2. Form input pengguna
# ---------------------------------------------------------
st.subheader("Input Penggunaan Smartphone Anda")

col1, col2, col3 = st.columns(3)

with col1:
    user_time = st.number_input("Total Screen Time (menit/hari)", min_value=0, value=120)

with col2:
    user_unlock = st.number_input("Jumlah Unlock per Hari", min_value=0, value=50)

with col3:
    user_app_use = st.number_input("Total Notifikasi Aplikasi", min_value=0, value=90)

# Data input pengguna dalam format dataframe
user_df = pd.DataFrame({
    "Daily_Screen_Time_Hours": [user_time / 60],   # menit → jam
    "Screen_Unlocks_Per_Day": [user_unlock],
    "App_Notifications_Received": [user_app_use]
})

# ---------------------------------------------------------
# 3. Tombol untuk menampilkan hasil
# ---------------------------------------------------------
st.markdown("---")

if st.button("Tampilkan"):
    # Menggabungkan dataset asli dengan data input pengguna
    combined = pd.concat([df_clean, user_df], ignore_index=True)

    # Normalisasi data
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(combined)

    # Proses klustering dengan K-Means
    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(scaled)

    # Cluster milik user ada pada elemen terakhir
    user_cluster = labels[-1]

    st.subheader("Hasil Klustering")
    st.write(f"Anda termasuk ke dalam Cluster **{user_cluster}**.")

    # Rekomendasi berdasarkan cluster
    recommendations = {
        0: "Penggunaan rendah. Istirahat setiap 60-90 menit untuk menjaga fokus dan kesehatan mata.",
        1: "Penggunaan sedang. Disarankan istirahat setiap 40-60 menit untuk mencegah fatigue.",
        2: "Penggunaan tinggi. Harus istirahat setiap 20-30 menit (micro-break) untuk mencegah stres mata dan ketegangan leher."
    }

    st.subheader("Rekomendasi")
    st.write(recommendations[user_cluster])

    # Menampilkan kembali data input user
    st.subheader("Data Input Anda")

    user_df_display = user_df.rename(columns={
        "Daily_Screen_Time_Hours": "Daily Screen Time (Jam)",
        "Screen_Unlocks_Per_Day": "Unlock Count",
        "App_Notifications_Received": "App Notifications"
    })

    st.write(f"""
    **Daily Screen Time (Jam)** : {user_df_display['Daily Screen Time (Jam)'][0]:.2f}  
    **Unlock Count** : {user_df_display['Unlock Count'][0]}  
    **App Notifications** : {user_df_display['App Notifications'][0]}
    """)

else:
    st.info("Tekan tombol Tampilkan untuk melihat hasil analisis.")

