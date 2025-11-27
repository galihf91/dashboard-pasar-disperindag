# app.py GABUNGAN
# ===============================
# 📈 Dashboard Harga & Prediksi
# 🏪 Dashboard Tera Ulang Pasar
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import math
import plotly.express as px
import plotly.graph_objects as go
import base64
from pathlib import Path
from io import StringIO
import re

import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import geopandas as gpd
from folium.map import CustomPane

# Import model ML (dari proyek harga-prediksi)
from models import train_lstm_for, forecast_lstm

# Import utilities (dari proyek harga-prediksi)
from utils import (
    clean_commodity_name,
    normalize_market_name,
    prepare_price_dataframe,
    format_rupiah,
    categorize_commodity,
    get_category_color,
    kebijakan_saran
)

# ----------------------------------------------------------
# KONFIGURASI HALAMAN UTAMA
# ----------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Pasar & Tera Ulang – Kab. Tangerang",
    page_icon="📊",
    layout="wide"
)

# Hilangkan menu & footer Streamlit
hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# ==========================================================
# =============== BAGIAN: DASHBOARD HARGA ==================
# ==========================================================

@st.cache_data
def load_harga_data():
    df_raw = pd.read_csv("harga_pasar_2024.csv")  # sesuaikan path jika perlu
    df_clean = prepare_price_dataframe(df_raw)
    return df_clean

def get_base64_of_image(image_path: str) -> str:
    """
    Mengubah file gambar lokal menjadi string base64
    agar bisa dipakai di CSS background-image.
    """
    img_path = Path(image_path)
    if not img_path.exists():
        return ""
    with open(img_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")

def get_komoditas_style(nama: str):
    """
    Mengembalikan (kategori, bg_color, badge_color) berdasarkan nama komoditas.
    """
    n = str(nama).lower()

    # Beras
    if "beras" in n:
        return "BERAS", "#FFF8E1", "#F9A825"  # bg kuning, badge kuning tua

    # Minyak goreng
    if "minyak" in n:
        return "MINYAK", "#FFF3E0", "#FB8C00"

    # Cabe / cabai
    if "cabe" in n or "cabai" in n or "rawit" in n:
        return "CABAI", "#FFEBEE", "#E53935"

    # Bawang
    if "bawang" in n:
        return "BAWANG", "#EDE7F6", "#8E24AA"

    # Tepung / terigu
    if "tepung" in n or "segitiga biru" in n:
        return "TEPUNG", "#E8F5E9", "#43A047"

    # Gula
    if "gula" in n:
        return "GULA", "#F3E5F5", "#7B1FA2"

    # Default / lain-lain
    return "PETERNAKAN", "#F5F5F5", "#757575"

def show_harga_prediksi_page():
    # ---------- CSS Kartu ----------
    st.markdown(
        """
        <style>
        .komod-card {
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .komod-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 4px 10px rgba(0,0,0,0.18);
        }
        .komod-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 999px;
            font-size: 10px;
            font-weight: 600;
            color: white;
            margin-left: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # CSS Background Ungu
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #F3E5F5 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Header dengan background foto (opsional)
    img_b64 = get_base64_of_image("assets/background_header.jpeg")
    if img_b64:
        st.markdown(
            f"""
            <style>
            .header-banner {{
                width: 100%;
                height: 280px;
                background-image: url("data:image/jpeg;base64,{img_b64}");
                background-size: cover;
                background-position: center -300px;
                background-repeat: no-repeat;
                border-radius: 12px;
                margin-bottom: 20px;
            }}
            </style>

            <div class="header-banner"></div>
            """,
            unsafe_allow_html=True
        )

    # Header utama
    st.markdown(
        """
        <div style='text-align:center; margin-bottom: 15px;'>
            <h1 style='margin-bottom: 0;'>Dashboard Harga Barang & Prediksi</h1>
            <p style='font-size:14px; margin-top:4px; color:#555;'>
                Dinas Perindustrian & Perdagangan Kabupaten Tangerang – Analisis Harga Pasar
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Load data harga
    try:
        df = load_harga_data()
    except Exception as e:
        st.error(f"Gagal membaca dataset 'harga_pasar_2024.csv': {e}")
        return

    # Init session state untuk model
    if "models" not in st.session_state:
        st.session_state["models"] = {}

    # Tab Layout
    tab1, tab2 = st.tabs(["📊 Harga Pasar", "🤖 Training & Prediksi"])

    # =========================================================
    # ========================= TAB 1 =========================
    # =========================================================
    with tab1:
        st.markdown("### 📊 Harga Komoditas per Pasar")

        pasar_list = sorted(df["pasar"].unique().tolist())
        pasar = st.selectbox("Pilih Pasar", pasar_list, key="pilih_pasar_tab1")

        # Filter data sesuai pasar terpilih
        df_pasar = df[df["pasar"] == pasar].copy()

        if df_pasar.empty:
            st.warning(f"Tidak ada data untuk pasar **{pasar}**.")
        else:
            df_pasar["tanggal"] = pd.to_datetime(df_pasar["tanggal"])
            min_date = df_pasar["tanggal"].min().date()
            max_date = df_pasar["tanggal"].max().date()

            # Layout utama: kiri (semua komoditas), kanan (detail komoditas)
            col_left, col_right = st.columns([2, 1])

            # ===================== KIRI: semua komoditas per tanggal =====================
            with col_left:
                st.markdown("#### 📅 Pilih Tanggal")

                selected_date = st.date_input(
                    "Tanggal harga yang ingin dilihat",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="tgl_pasar_kiri"
                )

                df_hari_ini = df_pasar[df_pasar["tanggal"].dt.date == selected_date].copy()

                if df_hari_ini.empty:
                    st.warning(f"Tidak ada data untuk pasar **{pasar}** pada tanggal **{selected_date}**.")
                else:
                    df_hari_ini = df_hari_ini.sort_values("komoditas")

                    st.markdown(f"#### 💰 Daftar Harga Komoditas di Pasar **{pasar}** pada {selected_date}")

                    # TAMPILAN KARTU KOMODITAS
                    num_cols = 3
                    cols = st.columns(num_cols)

                    for i, row in df_hari_ini.iterrows():
                        c = cols[i % num_cols]
                        nama = str(row["komoditas"])
                        harga = row["harga"]
                        kategori, bg_color, badge_color = get_komoditas_style(nama)

                        with c:
                            st.markdown(
                                f"""
                                <div class="komod-card" style="
                                    background-color: {bg_color};
                                    padding: 14px 16px;
                                    border-radius: 14px;
                                    margin-bottom: 12px;
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                    border: 1px solid rgba(0,0,0,0.08);
                                ">
                                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
                                        <div style="font-weight: 700; font-size: 14px;">
                                            {nama.upper()}
                                        </div>
                                        <span class="komod-badge" style="background-color: {badge_color};">
                                            {kategori}
                                        </span>
                                    </div>
                                    <div style="font-size: 12px; color: #555;">Harga</div>
                                    <div style="font-size: 20px; font-weight: 800; color: #1A237E;">
                                        Rp {harga:,.0f}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                    st.markdown("#### 📈 Grafik Harga per Komoditas")
                    df_plot = df_hari_ini.sort_values("komoditas").copy()

                    fig = px.bar(
                        df_plot,
                        x="komoditas",
                        y="harga",
                        text="harga",
                        color="harga",
                        color_continuous_scale="Blues",
                    )

                    fig.update_traces(
                        texttemplate="Rp %{y:,.0f}",
                        textposition="outside",
                        hovertemplate="<b>%{x}</b><br>Harga: Rp %{y:,.0f}<extra></extra>",
                    )

                    fig.update_layout(
                        title={
                            "text": f"Harga Komoditas di Pasar {pasar} pada {selected_date}",
                            "x": 0.5,
                            "xanchor": "center",
                            "font": {"size": 16},
                        },
                        xaxis_title="Komoditas",
                        yaxis_title="Harga (Rp)",
                        xaxis_tickangle=-60,
                        template="plotly_white",
                        coloraxis_showscale=False,
                        margin=dict(l=40, r=20, t=60, b=80),
                    )
                    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="rgba(0,0,0,0.15)")
                    st.plotly_chart(fig, use_container_width=True)

            # ===================== KANAN: detail riwayat satu komoditas =====================
            with col_right:
                st.markdown("#### 🔍 Detail Per Komoditas")

                komoditas_pasar_list = sorted(df_pasar["komoditas"].unique().tolist())

                selected_option = st.selectbox(
                    "Pilih komoditas",
                    ["— Pilih komoditas —"] + komoditas_pasar_list,
                    index=0,
                    key=f"detail_{pasar}"
                )

                if selected_option == "— Pilih komoditas —":
                    st.info("Pilih komoditas untuk melihat riwayat harganya.")
                else:
                    komoditas_detail = selected_option
                    df_view = df_pasar[df_pasar["komoditas"] == komoditas_detail].copy().sort_values("tanggal")

                    if df_view.empty:
                        st.warning(f"Tidak ada data untuk {komoditas_detail}.")
                    else:
                        df_view["tanggal"] = pd.to_datetime(df_view["tanggal"])

                        st.caption(
                            f"Periode: {df_view['tanggal'].min().date()} s.d. {df_view['tanggal'].max().date()}"
                        )

                        st.markdown("#### 📉 Grafik Riwayat Harga Komoditas")

                        df_plot = df_view.sort_values("tanggal").copy()

                        fig = px.line(
                            df_plot,
                            x="tanggal",
                            y="harga",
                            markers=True,
                            line_shape="spline",
                        )

                        fig.update_traces(
                            line=dict(color="#1A73E8", width=3),
                            marker=dict(size=2, color="#0D47A1"),
                            hovertemplate="<b>%{x}</b><br>Harga: <b>Rp %{y:,.0f}</b><extra></extra>",
                        )

                        fig.update_layout(
                            title={
                                "text": f"Riwayat Harga {komoditas_detail} - Pasar {pasar}",
                                "x": 0.5,
                                "xanchor": "center",
                                "font": {"size": 16},
                            },
                            xaxis_title="Tanggal",
                            yaxis_title="Harga (Rp)",
                            template="plotly_white",
                            hovermode="x unified",
                            margin=dict(l=40, r=20, t=50, b=40),
                            paper_bgcolor="white",
                            plot_bgcolor="rgba(230,242,255,1)"
                        )

                        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="rgba(0,0,0,0.15)")
                        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="rgba(0,0,0,0.10)")
                        st.plotly_chart(fig, use_container_width=True)

    # =========================================================
    # ========================= TAB 2 =========================
    # =========================================================
    with tab2:
        st.markdown("### 🤖 Training Model & Prediksi Harga + Saran Kebijakan")

        # Gunakan pasar terakhir yang dipilih di Tab 1
        # (variabel 'pasar' sudah didefinisikan di Tab 1)
        df_pasar = df[df["pasar"] == pasar].copy()

        if df_pasar.empty:
            st.warning(f"Tidak ada data untuk pasar **{pasar}**.")
        else:
            komoditas_list_pasar = sorted(df_pasar["komoditas"].unique().tolist())

            komoditas = st.selectbox(
                "Pilih Komoditas untuk Dilatih",
                komoditas_list_pasar,
                key="komoditas_pilih_tab2"
            )

            st.markdown("#### ⚙️ Pengaturan Model LSTM")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                window_size = st.slider("Window size (hari historis)", 7, 60, 30)
            with col_b:
                forecast_days = st.slider("Hari prediksi", 7, 60, 30)
            with col_c:
                epochs = st.slider("Epoch training", 5, 100, 30, step=5)

            df_sub = df[
                (df["komoditas"] == komoditas) &
                (df["pasar"] == pasar)
            ].sort_values("tanggal")

            if df_sub.empty:
                st.warning(f"Tidak ada data untuk komoditas **{komoditas}** di pasar **{pasar}**.")
                st.stop()

            st.markdown(
                f"**Kombinasi aktif:** {komoditas} – {pasar}  \n"
                f"Periode data: {df_sub['tanggal'].min().date()} s.d. {df_sub['tanggal'].max().date()}"
            )

            if "models" not in st.session_state:
                st.session_state["models"] = {}

            model_key = f"{komoditas}_{pasar}_{window_size}"
            model_obj = st.session_state["models"].get(model_key, None)

            train_button = st.button("🔁 Latih / Perbarui Model untuk Kombinasi Ini")

            if train_button:
                model, scaler, df_sub_trained, history, (mae, rmse) = train_lstm_for(
                    df,
                    komoditas=komoditas,
                    pasar=pasar,
                    window_size=window_size,
                    epochs=epochs
                )
                if model is not None:
                    st.session_state["models"][model_key] = {
                        "model": model,
                        "scaler": scaler,
                        "mae": mae,
                        "rmse": rmse,
                    }
                    model_obj = st.session_state["models"][model_key]
                    df_sub = df_sub_trained

            if model_obj is None:
                st.warning("Model belum dilatih. Tekan tombol di atas untuk melatih.")
            else:
                model = model_obj["model"]
                scaler = model_obj["scaler"]
                mae = model_obj["mae"]
                rmse = model_obj["rmse"]

                df_pred = forecast_lstm(
                    model,
                    scaler,
                    df_sub,
                    n_days=forecast_days,
                    window_size=window_size
                )

                st.markdown("#### 📋 Tabel Hasil Prediksi")

                df_pred_tampil = df_pred.copy()
                df_pred_tampil["tanggal"] = pd.to_datetime(df_pred_tampil["tanggal"])
                df_pred_tampil = df_pred_tampil.sort_values("tanggal").reset_index(drop=True)
                df_pred_tampil.insert(0, "Hari ke-", df_pred_tampil.index + 1)
                df_pred_tampil["prediksi"] = df_pred_tampil["prediksi"].round(0).astype(int)
                df_pred_tampil["tanggal"] = df_pred_tampil["tanggal"].dt.strftime("%d-%m-%Y")
                df_pred_tampil = df_pred_tampil.rename(columns={
                    "tanggal": "Tanggal",
                    "prediksi": "Prediksi Harga (Rp)"
                })

                st.markdown("""
                <style>
                .dataframe-container {
                    max-width: 600px;
                    margin: 0 auto;
                }
                </style>
                """, unsafe_allow_html=True)

                col1, col2, col3 = st.columns([0.25, 0.25, 0.25])
                with col2:
                    st.dataframe(
                        df_pred_tampil,
                        use_container_width=True,
                        hide_index=True
                    )

                st.markdown("#### 📈 Grafik Aktual vs Prediksi (Plotly Premium)")

                df_sub_plot = df_sub.copy()
                df_sub_plot["tanggal"] = pd.to_datetime(df_sub_plot["tanggal"])

                df_pred_plot = df_pred.copy()
                df_pred_plot["tanggal"] = pd.to_datetime(df_pred_plot["tanggal"])

                last_actual_date = df_sub_plot["tanggal"].max()

                fig = go.Figure()

                # Garis AKTUAL
                fig.add_trace(
                    go.Scatter(
                        x=df_sub_plot["tanggal"],
                        y=df_sub_plot["harga"],
                        mode="lines+markers",
                        name="Aktual",
                        line=dict(color="#1A73E8", width=3),
                        marker=dict(size=4),
                        hovertemplate="<b>%{x|%d-%m-%Y}</b><br>Aktual: <b>Rp %{y:,.0f}</b><extra></extra>",
                    )
                )

                # Garis PREDIKSI
                fig.add_trace(
                    go.Scatter(
                        x=df_pred_plot["tanggal"],
                        y=df_pred_plot["prediksi"],
                        mode="lines+markers",
                        name="Prediksi",
                        line=dict(color="#E53935", width=3, dash="dash"),
                        marker=dict(size=4),
                        fill="tozeroy",
                        fillcolor="rgba(229, 57, 53, 0.15)",
                        hovertemplate="<b>%{x|%d-%m-%Y}</b><br>Prediksi: <b>Rp %{y:,.0f}</b><extra></extra>",
                    )
                )

                fig.add_shape(
                    type="line",
                    x0=last_actual_date,
                    x1=last_actual_date,
                    y0=0,
                    y1=1,
                    xref="x",
                    yref="paper",
                    line=dict(color="gray", width=2, dash="dot"),
                )

                fig.add_annotation(
                    x=last_actual_date,
                    y=1,
                    xref="x",
                    yref="paper",
                    text="Mulai Prediksi",
                    showarrow=False,
                    font=dict(size=10, color="gray"),
                    yshift=10
                )

                fig.update_layout(
                    title={
                        "text": f"Grafik Aktual vs Prediksi\n{komoditas} – Pasar {pasar}",
                        "x": 0.5,
                        "xanchor": "center",
                        "font": {"size": 16},
                    },
                    xaxis_title="Tanggal",
                    yaxis_title="Harga (Rp)",
                    template="plotly_white",
                    hovermode="x unified",
                    paper_bgcolor="rgba(250,250,250,1)",
                    plot_bgcolor="rgba(250,250,250,1)",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5
                    ),
                    margin=dict(l=40, r=20, t=70, b=40),
                )

                fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="rgba(0,0,0,0.15)")
                fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="rgba(0,0,0,0.06)")

                st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### 📑 Saran Kebijakan")
                saran = kebijakan_saran(df_sub, df_pred)
                st.markdown(saran)

# ==========================================================
# =============== BAGIAN: DASHBOARD TERA ULANG =============
# ==========================================================

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())

def parse_coord(val):
    """Parse koordinat dari berbagai format"""
    try:
        if pd.isna(val) or val == "":
            return np.nan, np.nan
        
        s = str(val).strip()
        # Handle format: "-6.26435, 106.42592"
        if ',' in s:
            parts = [p.strip() for p in s.split(',')]
            if len(parts) >= 2:
                lat = float(parts[0])
                lon = float(parts[1])
                # Auto-swap jika format terbalik (lon, lat)
                if abs(lat) > 90 and abs(lon) <= 90:
                    lat, lon = lon, lat
                return lat, lon
    except Exception as e:
        st.warning(f"Gagal parse koordinat: {val}, error: {e}")
    return np.nan, np.nan

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    rename_mapping = {
        'Nama Pasar': 'nama_pasar',
        'Alamat': 'alamat',
        'Kecamatan': 'kecamatan', 
        'Koordinat': 'koordinat',
        'Tahun Tera Ulang': 'tera_ulang_tahun',
        'Total UTTP': 'jumlah_timbangan_tera_ulang',
        'Total Pedagang': 'total_pedagang'
    }
    
    existing_rename = {k: v for k, v in rename_mapping.items() if k in df.columns}
    df = df.rename(columns=existing_rename)
    
    if 'koordinat' in df.columns:
        coords = df['koordinat'].apply(parse_coord)
        df['lat'] = coords.apply(lambda x: x[0])
        df['lon'] = coords.apply(lambda x: x[1])
    
    timbangan_cols = ['Timb. Pegas', 'Timb. Meja', 'Timb. Elektronik', 
                      'Timb. Sentisimal', 'Timb. Bobot Ingsut', 'Neraca']
    
    available_timbangan_cols = [col for col in timbangan_cols if col in df.columns]
    
    if available_timbangan_cols:
        def summarize_timbangan(row):
            parts = []
            for col in available_timbangan_cols:
                try:
                    val = pd.to_numeric(row[col], errors='coerce')
                    if pd.notna(val) and val > 0:
                        label = col.replace('Timb. ', '').replace('Timb.', '')
                        parts.append(f"{label}: {int(val)}")
                except Exception:
                    continue
            return "; ".join(parts) if parts else "Tidak ada data"
        
        df['jenis_timbangan'] = df.apply(summarize_timbangan, axis=1)
    
    return df

@st.cache_data
def load_csv(path: str):
    import csv
    try:
        with open(path, "rb") as f:
            raw = f.read()
        text = raw.decode("utf-8-sig", errors="ignore")

        lines = text.splitlines()
        # Deteksi apakah mayoritas baris terkutip penuh
        if len(lines) > 1:
            quoted = sum(1 for l in lines[1:] if l.strip().startswith('"') and l.strip().endswith('"'))
            if quoted >= 0.8 * max(1, len(lines[1:])):
                fixed = [lines[0]]
                for line in lines[1:]:
                    s = line.strip()
                    if s.startswith('"') and s.endswith('"'):
                        s = s[1:-1].replace('""','"')
                    fixed.append(s)
                text = "\n".join(fixed)

        # Tentukan delimiter (fallback koma)
        try:
            sep = csv.Sniffer().sniff(lines[0]).delimiter
        except Exception:
            sep = ","

        df = pd.read_csv(StringIO(text), sep=sep)
        df = standardize_columns(df)
        return df, None

    except Exception as e:
        st.error(f"❌ Error loading CSV: {e}")
        return create_sample_data(), "Menggunakan data sample"

def create_sample_data():
    """Buat data sample jika file asli bermasalah"""
    st.warning("Membuat data sample karena file asli bermasalah")
    
    sample_data = {
        'nama_pasar': ['Cisoka', 'Curug', 'Mauk', 'Cikupa', 'Pasar Kemis'],
        'kecamatan': ['Cisoka', 'Curug', 'Mauk', 'Cikupa', 'Pasar Kemis'],
        'alamat': [
            'Jl. Ps. Cisoka No.44, Cisoka, Kec. Cisoka, Kabupaten Tangerang, Banten 15730',
            'Jl. Raya Curug, Curug Wetan, Kec. Curug, Kabupaten Tangerang, Banten 15810', 
            'East Mauk, Mauk, Tangerang Regency, Banten 15530',
            'Jl. Raya Serang, Cikupa, Kec. Cikupa, Kabupaten Tangerang, Banten 15710',
            'RGPJ+FJX, Jalan Raya, Sukaasih, Pasar Kemis, Tangerang Regency, Banten 15560'
        ],
        'lat': [-6.26435, -6.26100, -6.06044, -6.22907, -6.16365],
        'lon': [106.42592, 106.55858, 106.51129, 106.51981, 106.53155],
        'tera_ulang_tahun': [2025, 2025, 2025, 2025, 2025],
        'jumlah_timbangan_tera_ulang': [195, 251, 161, 257, 174],
        'jenis_timbangan': [
            'Pegas: 77; Meja: 30; Elektronik: 87',
            'Pegas: 60; Meja: 76; Elektronik: 107', 
            'Pegas: 80; Meja: 10; Elektronik: 71',
            'Pegas: 36; Meja: 88; Elektronik: 130',
            'Pegas: 54; Meja: 48; Elektronik: 72'
        ]
    }
    return pd.DataFrame(sample_data)

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Pastikan tipe data konsisten"""
    if 'tera_ulang_tahun' in df.columns:
        df['tera_ulang_tahun'] = pd.to_numeric(df['tera_ulang_tahun'], errors='coerce').fillna(0).astype(int)
    
    if 'jumlah_timbangan_tera_ulang' in df.columns:
        df['jumlah_timbangan_tera_ulang'] = pd.to_numeric(df['jumlah_timbangan_tera_ulang'], errors='coerce').fillna(0).astype(int)
    
    for col in ['nama_pasar', 'alamat', 'kecamatan', 'jenis_timbangan']:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    
    for c in ['lat', 'lon']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    
    return df

def clean_str_series(series: pd.Series) -> pd.Series:
    """Bersihkan series string"""
    if series is None:
        return pd.Series([], dtype=str)
    s = series.astype(str).str.strip()
    s = s.str.title()
    mask_bad = s.str.lower().isin(["", "nan", "none", "null", "na", "n/a", "-", "--"])
    return s[~mask_bad]

def uniq_clean(series: pd.Series) -> list:
    """Ambil nilai unik yang sudah dibersihkan"""
    return sorted(clean_str_series(series).unique().tolist())

def marker_color(year: int):
    """Tentukan warna marker berdasarkan tahun"""
    this_year = datetime.now().year
    if year is None or year == 0:
        return "gray"
    if year >= this_year:
        return "green"
    elif year == this_year - 1:
        return "orange"
    else:
        return "red"

def show_tera_ulang_page():
    st.title("🏪 Status Tera Ulang Timbangan Pasar – Kabupaten Tangerang")
    st.caption("Dinas Perindustrian dan Perdagangan • Bidang Kemetrologian")

    df, err = load_csv("DATA DASHBOARD PASAR.csv")
    if err:
        st.warning(f"Peringatan: {err}")

    df = coerce_types(df)

    # === SIDEBAR FILTERS ===
    with st.sidebar:
        st.header("Filter")
        mode = st.radio(
            "Mode pemilihan",
            options=["Pilih Kecamatan dulu → pilih Pasar", "Langsung pilih Pasar"],
            index=0
        )

        # Slider tahun
        if 'tera_ulang_tahun' in df.columns and df['tera_ulang_tahun'].notna().any():
            year_min = int(df['tera_ulang_tahun'].min())
            year_max = int(df['tera_ulang_tahun'].max())
        else:
            year_min = datetime.now().year - 5
            year_max = datetime.now().year

        if year_min == year_max:
            year_sel = (year_min, year_max)
            st.info(f"Data hanya punya satu tahun: {year_min}")
        else:
            year_sel = st.slider(
                "Rentang Tahun Tera Ulang",
                min_value=year_min,
                max_value=year_max,
                value=(year_min, year_max),
                step=1
            )

        all_kec = uniq_clean(df['kecamatan']) if 'kecamatan' in df.columns else []
        all_pasar = uniq_clean(df['nama_pasar']) if 'nama_pasar' in df.columns else []

        if mode.startswith("Pilih Kecamatan"):
            kec_opsi = ["(Semua)"] + all_kec
            kec = st.selectbox("Kecamatan", kec_opsi, index=0)

            if kec != "(Semua)" and {'kecamatan', 'nama_pasar'}.issubset(df.columns):
                pasar_opsi = ["(Semua)"] + uniq_clean(df.loc[df['kecamatan'] == kec, 'nama_pasar'])
            else:
                pasar_opsi = ["(Semua)"] + all_pasar

            nama_pasar = st.selectbox("Nama Pasar", pasar_opsi, index=0)
        else:
            kec = "(Semua)"
            pasar_opsi = ["(Semua)"] + all_pasar
            nama_pasar = st.selectbox("Nama Pasar", pasar_opsi, index=0)

        # Kartu info pasar terpilih
        if ('nama_pasar' in df.columns) and (nama_pasar != "(Semua)"):
            info = df.loc[df['nama_pasar'] == nama_pasar].head(1)
            if not info.empty:
                nama = info['nama_pasar'].iat[0]
                alamat = info['alamat'].iat[0] if 'alamat' in info.columns else "Alamat tidak tersedia"
                kecamatan = info['kecamatan'].iat[0] if 'kecamatan' in info.columns else "–"

                st.markdown("---")
                st.markdown(
                    f"""
                    <div style="
                        background-color:#f3e8ff;
                        padding:14px 16px;
                        border-radius:12px;
                        border-left:5px solid #8000FF;
                        box-shadow:0px 1px 4px rgba(0,0,0,0.15);
                        margin-top:10px;
                        ">
                        <h4 style="margin-bottom:6px; color:#4B0082; font-size:16px;">
                            🏪 {nama}
                        </h4>
                        <p style="margin:2px 0; font-size:13px; color:#222;">
                            <b>Kecamatan:</b> {kecamatan}<br>
                            <b>Alamat:</b> {alamat}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # Ringkasan Timbangan di Sidebar
        st.markdown("---")
        st.subheader("⚖️ Total Timbangan Tera Ulang")

        timb_map = {
            "Pegas":        ["Timb. Pegas", "Timb Pegas", "Pegas"],
            "Meja":         ["Timb. Meja", "Timb Meja", "Meja"],
            "Elektronik":   ["Timb. Elektronik", "Timb Elektronik", "Elektronik"],
            "Sentisimal":   ["Timb. Sentisimal", "Timb Sentisimal", "Sentisimal"],
            "Bobot Ingsut": ["Timb. Bobot Ingsut", "Timb Bobot Ingsut", "Bobot Ingsut"],
            "Neraca":       ["Neraca"]
        }

        def sum_first_existing(df_src: pd.DataFrame, candidates: list) -> int:
            for c in candidates:
                if c in df_src.columns:
                    return int(pd.to_numeric(df_src[c], errors="coerce").fillna(0).sum())
            return 0

        fdf_sidebar = df.copy()
        if 'tera_ulang_tahun' in fdf_sidebar.columns:
            fdf_sidebar = fdf_sidebar[(fdf_sidebar['tera_ulang_tahun'] >= year_sel[0]) &
                                      (fdf_sidebar['tera_ulang_tahun'] <= year_sel[1])]
        if 'kecamatan' in fdf_sidebar.columns and kec != "(Semua)":
            fdf_sidebar = fdf_sidebar[fdf_sidebar['kecamatan'] == kec]
        if 'nama_pasar' in fdf_sidebar.columns and nama_pasar != "(Semua)":
            fdf_sidebar = fdf_sidebar[fdf_sidebar['nama_pasar'] == nama_pasar]

        totals = {label: sum_first_existing(fdf_sidebar, cands) for label, cands in timb_map.items()}

        if 'jumlah_timbangan_tera_ulang' in fdf_sidebar.columns:
            total_uttp = int(pd.to_numeric(fdf_sidebar['jumlah_timbangan_tera_ulang'],
                                           errors='coerce').fillna(0).sum())
        else:
            total_uttp = sum(totals.values())

        for label, val in totals.items():
            st.markdown(f"**{label}**: {val:,}")

        st.markdown(f"**Total UTTP (semua jenis):** {total_uttp:,}")
        st.markdown("---")

    # FILTER DATA UTAMA
    fdf = df.copy()
    if 'tera_ulang_tahun' in fdf.columns:
        fdf = fdf[(fdf['tera_ulang_tahun'] >= year_sel[0]) & (fdf['tera_ulang_tahun'] <= year_sel[1])]
    if 'kecamatan' in fdf.columns and kec != "(Semua)":
        fdf = fdf[fdf['kecamatan'] == kec]
    if 'nama_pasar' in fdf.columns and nama_pasar != "(Semua)":
        fdf = fdf[fdf['nama_pasar'] == nama_pasar]

    # KPIs
    if kec == "(Semua)" and nama_pasar == "(Semua)":
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            total_kec = clean_str_series(df['kecamatan']).nunique() if 'kecamatan' in df.columns else 0
            st.metric("Total Kecamatan", total_kec)
        with c2:
            total_pasar = clean_str_series(df['nama_pasar']).nunique() if 'nama_pasar' in df.columns else 0
            st.metric("Total Seluruh Pasar", total_pasar)
        with c3:
            if 'tera_ulang_tahun' in fdf.columns and fdf['tera_ulang_tahun'].notna().any():
                latest_year = int(fdf['tera_ulang_tahun'].max())
            else:
                latest_year = "–"
            st.metric("Tahun Tera Ulang Terbaru", latest_year)
        with c4:
            total_timb = int(fdf['jumlah_timbangan_tera_ulang'].fillna(0).sum()) if 'jumlah_timbangan_tera_ulang' in fdf.columns else 0
            st.metric("Total Timbangan Tera Ulang", total_timb)
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            display_name = kec if kec != "(Semua)" else nama_pasar
            st.metric("Kecamatan Terpilih", display_name)
        with c2:
            if 'tera_ulang_tahun' in fdf.columns and fdf['tera_ulang_tahun'].notna().any():
                latest_year = int(fdf['tera_ulang_tahun'].max())
            else:
                latest_year = "–"
            st.metric("Tahun Tera Ulang Terbaru", latest_year)
        with c3:
            total_timb = int(fdf['jumlah_timbangan_tera_ulang'].fillna(0).sum()) if 'jumlah_timbangan_tera_ulang' in fdf.columns else 0
            st.metric("Total Timbangan Tera Ulang", total_timb)

    # MAP
    st.subheader("🗺️ Peta Lokasi Pasar")

    default_center = [-6.2, 106.55]
    default_zoom = 10

    has_coords = {'lat','lon'}.issubset(fdf.columns)
    coords = None
    if has_coords:
        try:
            coords = fdf[['lat','lon']].astype(float).dropna()
        except Exception as e:
            st.warning(f"Error processing coordinates: {e}")
            coords = None

    center_loc = default_center
    zoom_start = default_zoom

    if 'nama_pasar' in fdf.columns and nama_pasar != "(Semua)" and coords is not None and not coords.empty:
        row_sel = fdf[fdf['nama_pasar'] == nama_pasar]
        if not row_sel.empty:
            try:
                lat0 = float(row_sel['lat'].iloc[0])
                lon0 = float(row_sel['lon'].iloc[0])
                if pd.notna(lat0) and pd.notna(lon0):
                    center_loc = [lat0, lon0]
                    zoom_start = 16
            except Exception:
                pass
    elif coords is not None and len(coords) == 1:
        center_loc = [coords.iloc[0]['lat'], coords.iloc[0]['lon']]
        zoom_start = 14

    m = folium.Map(location=center_loc, zoom_start=zoom_start, control_scale=True, tiles="OpenStreetMap")

    # Pane untuk batas kecamatan
    m.add_child(CustomPane("batas-top", z_index=650))

    # Muat batas kecamatan dari GeoJSON
    batas = gpd.read_file("batas_kecamatan_tangerang.geojson")

    # Pastikan CRS ke WGS84 (lat-lon) untuk Folium
    if not batas.crs or batas.crs.to_epsg() != 4326:
        batas = batas.to_crs(epsg=4326)

    try:
        batas["geometry"] = batas.geometry.simplify(0.0005, preserve_topology=True)
    except Exception:
        pass

    kolom_nama = next(
        (c for c in ["NAMOBJ", "WADMKC", "KECAMATAN", "NAMA_KEC", "NAMA_KECAMATAN", "Kecamatan"]
         if c in batas.columns),
        None
    )

    tooltip_args = {}
    if kolom_nama:
        tooltip_args["tooltip"] = folium.GeoJsonTooltip(
            fields=[kolom_nama],
            aliases=["Kecamatan:"]
        )

    folium.GeoJson(
        batas,
        name="Batas Kecamatan",
        pane="batas-top",
        style_function=lambda x: {
            "color": "#8000FF",
            "weight": 2,
            "opacity": 1.0,
            "fill": False,
            "fillOpacity": 0,
        },
        **tooltip_args
    ).add_to(m)

    if has_coords and coords is not None and not coords.empty:
        cluster = MarkerCluster(name="Pasar", show=True).add_to(m)
        
        for _, r in fdf.iterrows():
            try:
                lat = float(r.get('lat', float('nan')))
                lon = float(r.get('lon', float('nan')))
            except Exception:
                lat, lon = float("nan"), float("nan")
            
            if pd.isna(lat) or pd.isna(lon):
                continue

            nama_p = str(r.get('nama_pasar', 'Unknown'))
            alamat_p = str(r.get('alamat', 'Tidak ada alamat'))
            tahun = r.get('tera_ulang_tahun', None)
            jumlah = r.get('jumlah_timbangan_tera_ulang', None)
            jenis = str(r.get('jenis_timbangan', 'Tidak ada data'))

            html = f"""
            <div style='width: 280px; font-family: Arial, sans-serif;'>
                <h4 style='margin:8px 0; color: #2E86AB;'>{nama_p}</h4>
                <div style='font-size: 12px; color:#666; margin-bottom:8px;'>{alamat_p}</div>
                <hr style='margin:6px 0'/>
                <table style='font-size: 12px; width: 100%;'>
                    <tr><td><b>Tera Ulang</b></td><td style='padding-left:8px'>: {tahun if pd.notna(tahun) else 'Tidak ada data'}</td></tr>
                    <tr><td><b>Jumlah Timbangan</b></td><td style='padding-left:8px'>: {jumlah if pd.notna(jumlah) else 'Tidak ada data'}</td></tr>
                    <tr><td><b>Jenis Timbangan</b></td><td style='padding-left:8px'>: {jenis}</td></tr>
                </table>
            </div>
            """
            
            tooltip_text = f"{nama_p} - {tahun if pd.notna(tahun) else 'Tahun tidak diketahui'}"
            popup = folium.Popup(html, max_width=320)
            tooltip = folium.Tooltip(tooltip_text)

            folium.CircleMarker(
                location=[lat, lon],
                radius=10,
                color=marker_color(int(tahun) if pd.notna(tahun) else None),
                fill=True,
                fill_color=marker_color(int(tahun) if pd.notna(tahun) else None),
                fill_opacity=0.7,
                weight=2,
                tooltip=tooltip,
                popup=popup
            ).add_to(cluster)

        if not ('nama_pasar' in fdf.columns and nama_pasar != "(Semua)") and len(coords) > 1:
            try:
                sw = [coords['lat'].min(), coords['lon'].min()]
                ne = [coords['lat'].max(), coords['lon'].max()]
                m.fit_bounds([sw, ne], padding=(30, 30))
            except Exception as e:
                st.warning(f"Tidak bisa auto-fit peta: {e}")
    else:
        st.warning("⚠️ Tidak ada data koordinat yang valid untuk ditampilkan di peta")

    st_folium(m, height=500, use_container_width=True)

# ==========================================================
# ===================== MENU UTAMA =========================
# ==========================================================

st.sidebar.title("📌 Menu Utama")
halaman = st.sidebar.radio(
    "Pilih Halaman",
    ["📈 Harga & Prediksi Komoditas", "🏪 Status Tera Ulang Pasar"]
)

if halaman == "📈 Harga & Prediksi Komoditas":
    show_harga_prediksi_page()
elif halaman == "🏪 Status Tera Ulang Pasar":
    show_tera_ulang_page()
