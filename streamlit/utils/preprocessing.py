"""
Preprocessing utilities for DBD Recommendation System.
Pipeline: cleaning -> waktu -> musim -> Rule-Based -> Target Prioritas -> TF-IDF+CART (rekomendasi).

Tidak ada SVM, dan sejak revisi ini TIDAK ADA LAGI TruncatedSVD. Alur sesuai
kode_final_skripsi.ipynb (versi revisi):
  Rule-Based Labelling -> Target Prioritas -> TF-IDF -> CART

TruncatedSVD dihapus dari pipeline training karena menyebabkan data leakage
(SVD di-fit menggunakan seluruh data, termasuk data uji, sebelum data displit).
Sebagai gantinya, setiap skenario (Semua Data / Musim Kemarau / Musim Hujan)
kini disimpan sebagai SATU sklearn.Pipeline yang menggabungkan TfidfVectorizer
+ DecisionTreeClassifier (CART). Prediksi cukup dilakukan dengan memanggil
`pipeline.predict(teks_bersih)` -- tidak perlu lagi transform TF-IDF/SVD manual.

Import di app.py:
    from utils.preprocessing import (
        clean_text, tentukan_prioritas_seri,
        pipeline_preprocessing, regenerate, baca_file_upload,
        LABEL_MAP, NORMALISASI_PROVINSI, BULAN_KEMARAU,
    )
"""

import re
import json
import numpy as np
import pandas as pd

# ============================================================
# KONSTANTA
# ============================================================

# Sesuai notebook: bulan kemarau = Mei-Oktober
BULAN_KEMARAU = [5, 6, 7, 8, 9, 10]

# Kelas CART — sesuai notebook (1/2/3, tanpa kelas 0)
LABEL_MAP = {
    1: 'PSN / 3M Plus',
    2: 'Fogging',
    3: 'Pemantauan Penanganan Medis',
}

NORMALISASI_PROVINSI = {
    'JAWA BARAT'                  : 'Jawa Barat',
    'JAWA TENGAH'                 : 'Jawa Tengah',
    'JAWA TIMUR'                  : 'Jawa Timur',
    'DKI JAKARTA'                 : 'DKI Jakarta',
    'BANTEN'                      : 'Banten',
    'DI YOGYAKARTA'               : 'DI Yogyakarta',
    'DIY YOGYAKARTA'              : 'DI Yogyakarta',
    'DIY'                         : 'DI Yogyakarta',
    'JOGJA'                       : 'DI Yogyakarta',
    'JOGJAKARTA'                  : 'DI Yogyakarta',
    'YOGYAKARTA'                  : 'DI Yogyakarta',
    'D.I. YOGYAKARTA'             : 'DI Yogyakarta',
    'D.I YOGYAKARTA'              : 'DI Yogyakarta',
    'DAERAH ISTIMEWA YOGYAKARTA'  : 'DI Yogyakarta',
    'SPECIAL REGION OF YOGYAKARTA': 'DI Yogyakarta',
    'SUMATERA UTARA'              : 'Sumatera Utara',
    'SUMATERA BARAT'              : 'Sumatera Barat',
    'SUMATERA SELATAN'            : 'Sumatera Selatan',
    'KALIMANTAN BARAT'            : 'Kalimantan Barat',
    'KALIMANTAN TENGAH'           : 'Kalimantan Tengah',
    'KALIMANTAN SELATAN'          : 'Kalimantan Selatan',
    'KALIMANTAN TIMUR'            : 'Kalimantan Timur',
    'SULAWESI UTARA'              : 'Sulawesi Utara',
    'SULAWESI TENGAH'             : 'Sulawesi Tengah',
    'SULAWESI SELATAN'            : 'Sulawesi Selatan',
    'SULAWESI TENGGARA'           : 'Sulawesi Tenggara',
    'NUSA TENGGARA BARAT'         : 'Nusa Tenggara Barat',
    'NUSA TENGGARA TIMUR'         : 'Nusa Tenggara Timur',
    'PAPUA BARAT'                 : 'Papua Barat',
    'PAPUA'                       : 'Papua',
    'MALUKU'                      : 'Maluku',
    'MALUKU UTARA'                : 'Maluku Utara',
    'ACEH'                        : 'Aceh',
    'RIAU'                        : 'Riau',
    'KEPULAUAN RIAU'              : 'Kepulauan Riau',
    'JAMBI'                       : 'Jambi',
    'BENGKULU'                    : 'Bengkulu',
    'LAMPUNG'                     : 'Lampung',
    'BANGKA BELITUNG'             : 'Bangka Belitung',
    'KEPULAUAN BANGKA BELITUNG'   : 'Bangka Belitung',
    'GORONTALO'                   : 'Gorontalo',
    'KALIMANTAN UTARA'            : 'Kalimantan Utara',
}

# Keyword rule-based — sama persis dengan notebook
_KW_GEJALA     = ["dbd","demam berdarah","demam","dengue","trombosit",
                   "panas tinggi","gejala","meninggal"]
_KW_NYAMUK     = ["nyamuk","jentik","aedes","aegypti","wolbachia","fogging",
                   "psn","jumantik","malaria"]
_KW_LINGKUNGAN = ["genangan","banjir","selokan","got","sampah","kotor",
                   "kumuh","air tergenang"]

_TWITTER_FORMATS = [
    '%a %b %d %H:%M:%S %z %Y',
    '%Y-%m-%dT%H:%M:%S%z',
    '%Y-%m-%d %H:%M:%S',
    '%Y-%m-%d',
]


# ============================================================
# FUNGSI DASAR
# ============================================================

def clean_text(text: str) -> str:
    """Bersihkan teks sesuai pipeline notebook."""
    text = str(text).lower()
    text = re.sub(r'http\S+',     ' ', text)
    text = re.sub(r'www\S+',      ' ', text)
    text = re.sub(r'@\w+',        ' ', text)
    text = re.sub(r'#\w+',        ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\d+',         ' ', text)
    text = re.sub(r'\s+',         ' ', text)
    return text.strip()


def _rule_based_label(text: str):
    """Rule-based labelling — sesuai fungsi di notebook."""
    t = str(text).lower()
    g = 1 if any(k in t for k in _KW_GEJALA)     else 0
    n = 1 if any(k in t for k in _KW_NYAMUK)     else 0
    l = 1 if any(k in t for k in _KW_LINGKUNGAN) else 0
    return g, n, l


def tentukan_prioritas_seri(g: int, n: int, l: int) -> int:
    """
    Konversi 3 label biner ke 1 target prioritas.
    Sesuai fungsi tentukan_target_prioritas_seri() di notebook.
    """
    total = g + n + l
    if total == 0:
        return 1
    elif total == 3:
        return 3
    elif g == 1 and n == 1:
        return 3 if l == 0 else 2
    elif g == 1:
        return 3
    elif n == 1:
        return 2
    elif l == 1:
        return 1
    return 1


def _parse_datetime(series: pd.Series) -> pd.Series:
    today  = pd.Timestamp('today').normalize()
    parsed = None
    for fmt in _TWITTER_FORMATS:
        try:
            parsed = pd.to_datetime(series, format=fmt, errors='coerce', utc=True)
            if parsed.notna().mean() > 0.5:
                break
        except Exception:
            continue
    if parsed is None or parsed.notna().mean() < 0.1:
        parsed = pd.to_datetime(series, errors='coerce', utc=True)
    parsed = parsed.fillna(pd.Timestamp(today, tz='UTC'))
    return parsed


def _deteksi_kolom(df: pd.DataFrame, kandidat: list):
    for k in kandidat:
        if k in df.columns:
            return k
    return None


def _normalisasi_provinsi(df: pd.DataFrame) -> pd.DataFrame:
    if 'lokasi_hasil_mining' in df.columns:
        df['provinsi_map'] = (
            df['lokasi_hasil_mining'].astype(str).str.upper().str.strip()
        )
    return df


def _prediksi_pipeline(pipeline, teks_bersih: list) -> np.ndarray:
    """
    Prediksi target_prioritas langsung dari teks bersih menggunakan
    SATU pipeline (TfidfVectorizer + DecisionTreeClassifier).

    Menggantikan alur lama (tfidf.transform -> svd.transform -> cart.predict)
    yang rentan data leakage karena SVD di-fit di luar konteks train/test.
    Pipeline sudah di-fit sepenuhnya saat training (lihat notebook), sehingga
    di sini hanya perlu `.predict()`.
    """
    return pipeline.predict(teks_bersih).astype(int)


# ============================================================
# PIPELINE UTAMA
# Pipeline: cleaning -> waktu -> musim -> Rule-Based ->
#           Target Prioritas -> TF-IDF+CART (satu Pipeline, tanpa SVD)
# Tidak ada SVM — sesuai kode_final_skripsi.ipynb (versi revisi)
# ============================================================

def pipeline_preprocessing(df_raw, pipeline) -> tuple:
    """
    Pipeline lengkap dari data mentah ke rekomendasi penanganan.

    Parameters
    ----------
    df_raw   : DataFrame mentah hasil upload
    pipeline : sklearn.pipeline.Pipeline (step 'tfidf' + 'cart'), di-load dari
               pipeline_semua.pkl / pipeline_kemarau.pkl / pipeline_hujan.pkl
               (satu file per skenario, hasil ekspor notebook training terbaru
               yang sudah tidak lagi memakai TruncatedSVD).

    Returns (df_hasil, laporan, None) atau (None, None, pesan_error).
    """
    df = df_raw.copy()

    kolom_teks   = _deteksi_kolom(df, ['full_text','text','tweet','content'])
    kolom_lokasi = _deteksi_kolom(df, ['lokasi_hasil_mining','lokasi','location','province'])
    kolom_waktu  = _deteksi_kolom(df, ['created_at','date','waktu','timestamp'])
    kolom_id     = _deteksi_kolom(df, ['id_str','id','tweet_id'])

    if kolom_teks is None:
        return None, None, (
            "Kolom teks tidak ditemukan. "
            "Pastikan ada kolom: full_text / text / tweet / content"
        )

    n_awal = len(df)

    # 1. Hapus baris kosong
    df       = df[df[kolom_teks].notna()].copy()
    n_kosong = n_awal - len(df)

    # 2. Hapus duplikat
    subset_dup = [kolom_id] if kolom_id else [kolom_teks]
    df         = df.drop_duplicates(subset=subset_dup).copy()
    n_duplikat = (n_awal - n_kosong) - len(df)
    df         = df.reset_index(drop=True)

    # 3. Teks asli + cleaning
    df['teks_asli']  = df[kolom_teks].astype(str)
    df['clean_text'] = df[kolom_teks].apply(clean_text)

    # 4. Waktu → bulan → musim
    if kolom_waktu:
        df['created_at']  = _parse_datetime(df[kolom_waktu].astype(str))
        df['bulan_nama']  = df['created_at'].dt.strftime('%b')
        df['bulan_angka'] = df['created_at'].dt.month.astype(int)
        df['musim']       = df['bulan_angka'].apply(
            lambda b: 'Kemarau' if b in BULAN_KEMARAU
                      else ('Hujan' if b > 0 else 'Tidak Diketahui')
        )
    else:
        df['created_at']  = pd.NaT
        df['bulan_nama']  = '-'
        df['bulan_angka'] = 0
        df['musim']       = 'Tidak Diketahui'

    # 5. Lokasi & normalisasi provinsi
    df['lokasi_hasil_mining'] = df[kolom_lokasi].astype(str) if kolom_lokasi else '-'
    df = _normalisasi_provinsi(df)

    teks_bersih = df['clean_text'].tolist()

    # 6. Rule-Based Labelling
    labels = df['clean_text'].apply(
        lambda t: pd.Series(_rule_based_label(t),
                            index=['gejala_berat','ada_nyamuk','risiko_lingkungan'])
    )
    df[['gejala_berat','ada_nyamuk','risiko_lingkungan']] = labels

    # 7. Target Prioritas dari Rule-Based
    df['target_prioritas'] = df.apply(
        lambda r: tentukan_prioritas_seri(
            int(r['gejala_berat']),
            int(r['ada_nyamuk']),
            int(r['risiko_lingkungan']),
        ), axis=1
    )

    # 8. TF-IDF + CART (satu Pipeline, tanpa SVD) → rekomendasi final
    y_cart = _prediksi_pipeline(pipeline, teks_bersih)
    df['target_prioritas'] = y_cart
    df['penanganan_utama'] = df['target_prioritas'].map(LABEL_MAP)

    kolom_final = [
        'teks_asli','clean_text','lokasi_hasil_mining','created_at',
        'bulan_nama','bulan_angka','musim','provinsi_map',
        'gejala_berat','ada_nyamuk','risiko_lingkungan',
        'target_prioritas','penanganan_utama',
    ]
    kolom_ada = [c for c in kolom_final if c in df.columns]
    laporan   = {
        'n_awal'    : n_awal,
        'n_kosong'  : n_kosong,
        'n_duplikat': n_duplikat,
        'n_bersih'  : len(df),
    }
    return df[kolom_ada].reset_index(drop=True), laporan, None


# ============================================================
# REGENERATE — setelah hapus baris manual
# ============================================================

def regenerate(df_aktif, pipeline) -> pd.DataFrame:
    """
    Hitung ulang label dan rekomendasi dari data yang tersisa.
    Rule-Based dijalankan ulang, lalu TF-IDF+CART (satu Pipeline, tanpa SVD).

    Parameters
    ----------
    df_aktif : DataFrame data yang masih aktif (sudah punya kolom clean_text)
    pipeline : sklearn.pipeline.Pipeline untuk skenario terkait
               (pipeline_semua.pkl / pipeline_kemarau.pkl / pipeline_hujan.pkl)
    """
    df = df_aktif.copy()
    teks_bersih = df['clean_text'].tolist()

    # Rule-Based ulang
    labels = df['clean_text'].apply(
        lambda t: pd.Series(_rule_based_label(t),
                            index=['gejala_berat','ada_nyamuk','risiko_lingkungan'])
    )
    df[['gejala_berat','ada_nyamuk','risiko_lingkungan']] = labels

    df['target_prioritas'] = df.apply(
        lambda r: tentukan_prioritas_seri(
            int(r['gejala_berat']),
            int(r['ada_nyamuk']),
            int(r['risiko_lingkungan']),
        ), axis=1
    )

    # TF-IDF + CART (satu Pipeline, tanpa SVD)
    y_cart = _prediksi_pipeline(pipeline, teks_bersih)
    df['target_prioritas'] = y_cart
    df['penanganan_utama'] = df['target_prioritas'].map(LABEL_MAP)

    return df.reset_index(drop=True)


# ============================================================
# BACA FILE UPLOAD
# ============================================================

def baca_file_upload(uploaded_file) -> tuple:
    """Baca file upload (CSV / Excel / JSON) menjadi DataFrame."""
    nama = uploaded_file.name.lower()
    try:
        if nama.endswith('.csv'):
            return pd.read_csv(uploaded_file), None
        elif nama.endswith(('.xlsx','.xls')):
            return pd.read_excel(uploaded_file), None
        elif nama.endswith('.json'):
            data = json.load(uploaded_file)
            if isinstance(data, list):
                return pd.DataFrame(data), None
            elif isinstance(data, dict):
                return pd.DataFrame.from_dict(data), None
            else:
                return None, "Format JSON tidak dikenali."
        else:
            return None, "Format tidak didukung. Gunakan CSV, Excel, atau JSON."
    except Exception as e:
        return None, f"Gagal membaca file: {e}"