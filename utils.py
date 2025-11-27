# utils.py
"""
Kumpulan fungsi utilitas untuk proyek:
- Dashboard Harga Barang & Prediksi
- Data: harga komoditas per pasar (tanggal, komoditas, pasar, harga)

Dipakai di app.py dengan:
    from utils import (
        clean_commodity_name,
        normalize_market_name,
        prepare_price_dataframe,
        format_rupiah,
        categorize_commodity,
        get_category_color,
    )
"""

from typing import Optional
import pandas as pd


# ---------------------------------------------------------
# 1. Pembersihan & standarisasi komoditas / pasar
# ---------------------------------------------------------

def clean_commodity_name(name: str) -> str:
    """
    Standarisasi nama komoditas berdasarkan mapping yang sudah disepakati.
    Contoh:
        CURAH -> MINYAK GORENG CURAH
        KEMASAN -> MINYAK GORENG KEMASAN
        MERAH BESAR -> CABE MERAH BESAR
        MERAH KERITING -> CABE MERAH KERITING
        MINYAK KITA -> MINYAK GORENG MINYAK KITA
        RAWIT HIJAU -> CABE RAWIT HIJAU
        RAWIT MERAH -> CABE RAWIT MERAH
        SEGITIGA BIRU (KW MEDIUM) -> TEPUNG SEGITIGA BIRU (KW MEDIUM)
    """
    if name is None:
        return ""

    raw = str(name).strip().upper()

    mapping = {
        "CURAH": "MINYAK GORENG CURAH",
        "KEMASAN": "MINYAK GORENG KEMASAN",
        "MERAH BESAR": "CABE MERAH BESAR",
        "MERAH KERITING": "CABE MERAH KERITING",
        "MINYAK KITA": "MINYAK GORENG MINYAK KITA",
        "RAWIT HIJAU": "CABE RAWIT HIJAU",
        "RAWIT MERAH": "CABE RAWIT MERAH",
        "SEGITIGA BIRU (KW MEDIUM)": "TEPUNG SEGITIGA BIRU (KW MEDIUM)",
    }

    return mapping.get(raw, raw)


def normalize_market_name(pasar: str) -> str:
    """
    Standarisasi nama pasar.
    Misal variasi:
        'PASAR CISOKA', 'CISOKA ' -> 'CISOKA'
        'PASAR SEPATAN', 'SEPATAN ' -> 'SEPATAN'
    Kalau tidak dikenali, dikembalikan dalam bentuk uppercase strip.
    """
    if pasar is None:
        return ""

    raw = str(pasar).strip().upper()

    mapping = {
        "PASAR CISOKA": "CISOKA",
        "CISOKA": "CISOKA",
        "PASAR SEPATAN": "SEPATAN",
        "SEPATAN": "SEPATAN",
    }

    return mapping.get(raw, raw)


def prepare_price_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membersihkan dan menyiapkan dataframe harga.
    Diasumsikan struktur long:
        kolom minimal: ['tanggal', 'komoditas', 'pasar', 'harga']
    Fungsi ini akan:
    - Menormalkan nama kolom (lowercase)
    - Konversi tanggal ke datetime
    - Uppercase & standarisasi nama komoditas & pasar
    - Drop baris tanpa tanggal / harga
    - Sort berdasarkan tanggal, komoditas, pasar
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["tanggal", "komoditas", "pasar", "harga"])

    # Normalisasi nama kolom ke lowercase
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # Coba mapping beberapa kemungkinan nama kolom
    col_map = {}
    if "tanggal" in df.columns:
        col_map["tanggal"] = "tanggal"
    elif "tgl" in df.columns:
        col_map["tgl"] = "tanggal"

    if "komoditas" in df.columns:
        col_map["komoditas"] = "komoditas"
    elif "komoditi" in df.columns:
        col_map["komoditi"] = "komoditas"

    if "pasar" in df.columns:
        col_map["pasar"] = "pasar"

    if "harga" in df.columns:
        col_map["harga"] = "harga"

    df = df.rename(columns=col_map)

    required = {"tanggal", "komoditas", "pasar", "harga"}
    missing = required - set(df.columns)
    if missing:
        # Kalau ada kolom wajib yang hilang, kembalikan df kosong
        return pd.DataFrame(columns=["tanggal", "komoditas", "pasar", "harga"])

    # Konversi tanggal
    df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")

    # Standarisasi komoditas & pasar
    df["komoditas"] = (
        df["komoditas"]
        .astype(str)
        .str.strip()
        .str.upper()
        .apply(clean_commodity_name)
    )
    df["pasar"] = (
        df["pasar"]
        .astype(str)
        .str.strip()
        .str.upper()
        .apply(normalize_market_name)
    )

    # Harga numeric
    df["harga"] = pd.to_numeric(df["harga"], errors="coerce")

    # Drop baris tidak valid
    df = df.dropna(subset=["tanggal", "harga"])

    # Sort
    df = df.sort_values(["tanggal", "komoditas", "pasar"]).reset_index(drop=True)

    return df[["tanggal", "komoditas", "pasar", "harga"]]


# ---------------------------------------------------------
# 2. Format tampilan (Rupiah, kategori, warna)
# ---------------------------------------------------------

def format_rupiah(value: Optional[float]) -> str:
    """
    Format angka menjadi teks Rupiah:
        29625 -> 'Rp 29.625'
    Jika None atau NaN, dikembalikan '-'.
    """
    if value is None:
        return "-"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"Rp {v:,.0f}"


def categorize_commodity(komoditas: str) -> str:
    """
    Mengelompokkan komoditas ke kategori besar untuk keperluan warna / badge.
    Contoh kategori:
      - 'BERAS'
      - 'MINYAK GORENG'
      - 'CABAI'
      - 'BAWANG'
      - 'GULA'
      - 'TELUR'
      - 'LAINNYA'
    """
    if komoditas is None:
        return "LAINNYA"

    name = str(komoditas).upper()

    if "BERAS" in name:
        return "BERAS"
    if "MINYAK" in name:
        return "MINYAK GORENG"
    if "CABAI" in name or "CABE" in name or "RAWIT" in name:
        return "CABAI"
    if "BAWANG" in name:
        return "BAWANG"
    if "GULA" in name:
        return "GULA"
    if "TELUR" in name:
        return "TELUR"
    if "AYAM" in name or "DAGING" in name:
        return "PROTEIN HEWANI"
    if "TEPUNG" in name:
        return "TEPUNG"
    return "LAINNYA"


def get_category_color(category_or_komoditas: str) -> str:
    """
    Mengembalikan kode warna (hex) untuk kategori tertentu.
    Bisa diberi input kategori langsung ('BERAS', 'CABAI', ...) atau langsung nama komoditas.
    """
    if category_or_komoditas is None:
        return "#3949AB"

    # Jika yang masuk ternyata nama komoditas, kategorikan dulu
    cat = categorize_commodity(category_or_komoditas)

    color_map = {
        "BERAS": "#1E88E5",
        "MINYAK GORENG": "#F9A825",
        "CABAI": "#E53935",
        "BAWANG": "#8E24AA",
        "GULA": "#6D4C41",
        "TELUR": "#FB8C00",
        "PROTEIN HEWANI": "#5E35B1",
        "TEPUNG": "#00897B",
        "LAINNYA": "#3949AB",
    }

    return color_map.get(cat, "#3949AB")

def kebijakan_saran(df_hist, df_pred, horizon_analisis: int = 7) -> str:
    """
    Menyusun saran kebijakan berbasis:
    - Harga aktual terakhir
    - Rata-rata prediksi beberapa hari ke depan (horizon_analisis)
    - Persentase perubahan (tren naik/turun/stabil)
    - Volatilitas (bergejolak atau stabil)

    df_hist : Data historis untuk 1 komoditas & 1 pasar
              kolom minimal: ['tanggal', 'komoditas', 'pasar', 'harga']
    df_pred : Dataframe hasil forecast_lstm
              kolom minimal: ['tanggal', 'prediksi']
    """
    import numpy as np  # just in case

    if df_hist is None or df_hist.empty or df_pred is None or df_pred.empty:
        return (
            "Data historis atau data prediksi belum tersedia sehingga "
            "belum dapat disusun saran kebijakan yang spesifik."
        )

    df_hist = df_hist.sort_values("tanggal").copy()
    df_pred = df_pred.sort_values("tanggal").copy()

    # Info komoditas & pasar (kalau ada)
    komoditas = df_hist["komoditas"].iloc[-1] if "komoditas" in df_hist.columns else "-"
    pasar = df_hist["pasar"].iloc[-1] if "pasar" in df_hist.columns else "-"

    # Harga aktual terakhir
    last_actual_price = float(df_hist["harga"].iloc[-1])

    # Ambil horizon analisis (misal 7 hari prediksi ke depan atau kurang jika datanya sedikit)
    h = min(horizon_analisis, len(df_pred))
    next_pred = df_pred.iloc[:h].copy()
    mean_pred = float(next_pred["prediksi"].mean())

    # Persen perubahan rata-rata prediksi vs harga terakhir aktual
    if last_actual_price > 0:
        change_pct = (mean_pred - last_actual_price) / last_actual_price * 100
    else:
        change_pct = 0.0

    # Volatilitas prediksi (rata-rata perubahan persen harian absolut)
    if len(df_pred) > 1:
        pct_changes = df_pred["prediksi"].pct_change().dropna().abs() * 100
        volatility = float(pct_changes.mean())
    else:
        volatility = 0.0

    # Klasifikasi tren
    if change_pct > 10:
        tren = "naik tajam"
    elif change_pct > 5:
        tren = "cenderung naik"
    elif change_pct < -10:
        tren = "turun tajam"
    elif change_pct < -5:
        tren = "cenderung turun"
    else:
        tren = "relatif stabil"

    # Klasifikasi volatilitas
    if volatility > 8:
        vol_text = "sangat bergejolak"
    elif volatility > 4:
        vol_text = "cukup bergejolak"
    else:
        vol_text = "relatif stabil"

    # Format angka untuk ditampilkan
    def fmt_rp(x: float) -> str:
        return f"Rp {x:,.0f}"

    change_dir = "lebih tinggi" if change_pct >= 0 else "lebih rendah"

    teks = []
    teks.append(f"**Ringkasan Prediksi Harga {komoditas} – Pasar {pasar}**")
    teks.append(f"- Harga aktual terakhir : **{fmt_rp(last_actual_price)}**")
    teks.append(
        f"- Rata-rata prediksi {h} hari ke depan : **{fmt_rp(mean_pred)}** "
        f"({abs(change_pct):.1f}% {change_dir} dibanding harga terakhir; tren **{tren}**)"
    )
    teks.append(
        f"- Pola pergerakan prediksi dikategorikan sebagai **{vol_text}** "
        f"(volatilitas sekitar {volatility:.1f}% per hari)."
    )

    teks.append("")
    teks.append("**Implikasi Kebijakan yang Disarankan:**")

    if "naik" in tren:
        teks.append(
            "- **Penguatan pasokan:** koordinasi dengan pemasok/gapoktan untuk "
            "meningkatkan pasokan ke pasar, terutama pada hari-hari dengan puncak permintaan."
        )
        teks.append(
            "- **Pantau potensi spekulasi harga:** lakukan sidak lapangan jika kenaikan dirasa "
            "tidak wajar untuk mencegah penahanan barang (stockpiling)."
        )
        teks.append(
            "- **Informasi harga ke masyarakat:** perkuat publikasi harga referensi agar konsumen "
            "memiliki acuan dan pedagang tidak menaikkan harga berlebihan."
        )
    elif "turun" in tren:
        teks.append(
            "- **Jaga agar penurunan harga tetap wajar:** pastikan penurunan tidak karena "
            "kualitas barang yang memburuk atau pasokan yang tidak terserap."
        )
        teks.append(
            "- **Dukung stabilisasi pendapatan pedagang/petani:** bila penurunan sangat tajam, "
            "dipertimbangkan intervensi seperti promosi pasar, operasi pembelian, atau kerjasama "
            "penyaluran ke pasar lain."
        )
    else:  # relatif stabil
        teks.append(
            "- **Lanjutkan pola distribusi saat ini:** karena harga relatif stabil, pola distribusi "
            "dan pasokan yang ada dapat dipertahankan dengan pemantauan rutin."
        )
        teks.append(
            "- **Fokus pada pemeliharaan kualitas dan kontinuitas pasokan** agar stabilitas harga "
            "dapat dipertahankan dalam jangka menengah."
        )

    # Tambahan bila volatil
    if "bergejolak" in vol_text:
        teks.append(
            "- **Perlu pemantauan lebih sering:** karena harga bergejolak, disarankan pemantauan harian "
            "dan koordinasi lintas pasar untuk mengantisipasi lonjakan mendadak."
        )

    return "\n".join(teks)
