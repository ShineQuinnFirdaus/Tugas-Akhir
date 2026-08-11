import os
import pandas as pd

# Jalur absolut ke folder root proyek
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Nama kelas sesuai LABEL_MAP di preprocessing.py
LABEL_MAP_INV = {
    1: 'PSN / 3M Plus',
    2: 'Fogging',
    3: 'Pemantauan Penanganan Medis',
}

WARNA_MAP = {
    'PSN / 3M Plus'               : {'warna': 'Hijau',  'hex': '#82b366'},
    'Fogging'                     : {'warna': 'Kuning', 'hex': '#d6a520'},
    'Pemantauan Penanganan Medis' : {'warna': 'Merah',  'hex': '#d63b3b'},
}

PROVINSI_VALID = [
    'Jawa Timur', 'Jawa Barat', 'Jawa Tengah',
    'DI Yogyakarta', 'DKI Jakarta', 'Banten',
]


def predict_from_pipeline(texts, pipeline_model) -> list:
    """
    Prediksi kelas penanganan langsung dari teks bersih (clean_text).

    Sejak TruncatedSVD dihapus dari pipeline pemodelan (karena menyebabkan
    data leakage), setiap skenario kini disimpan sebagai SATU objek
    sklearn.Pipeline yang sudah menggabungkan TfidfVectorizer + CART.
    Prediksi cukup dilakukan dengan memanggil `.predict()` langsung pada
    teks mentah/bersih -- tidak perlu lagi transformasi TF-IDF atau SVD
    manual sebelumnya.

    Parameters
    ----------
    texts          : list[str] atau pandas.Series berisi teks (clean_text)
    pipeline_model : sklearn.pipeline.Pipeline hasil joblib.load(...)
                      (mis. pipeline_semua.pkl / pipeline_kemarau.pkl /
                       pipeline_hujan.pkl), berisi step 'tfidf' + 'cart'.

    Returns
    -------
    list of int (1/2/3), sesuai kode pada LABEL_MAP_INV.
    """
    return pipeline_model.predict(texts).tolist()


def agregasi_per_provinsi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung rekomendasi dominan per provinsi berdasarkan kolom
    provinsi_map dan penanganan_utama.

    Parameters
    ----------
    df : DataFrame hasil pipeline_preprocessing (sudah punya provinsi_map & penanganan_utama)
    """
    hasil = []
    for prov in PROVINSI_VALID:
        sub = df[df['provinsi_map'] == prov]
        if len(sub) == 0:
            continue
        cnt    = sub['penanganan_utama'].value_counts()
        modus  = cnt.idxmax()
        info   = WARNA_MAP.get(modus, {'warna': '-', 'hex': '#cccccc'})
        hasil.append({
            'Provinsi'                   : prov,
            'Total Tweet'                : len(sub),
            'PSN / 3M Plus'              : cnt.get('PSN / 3M Plus',              0),
            'Fogging'                    : cnt.get('Fogging',                    0),
            'Pemantauan Penanganan Medis': cnt.get('Pemantauan Penanganan Medis', 0),
            'Rekomendasi'                : modus,
            'Warna'                      : info['warna'],
            'Hex'                        : info['hex'],
        })
    return pd.DataFrame(hasil)