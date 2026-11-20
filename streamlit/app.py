import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import os
import io

from utils.preprocessing import (
    pipeline_preprocessing,
    regenerate,
    baca_file_upload,
    NORMALISASI_PROVINSI,
)

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Sistem Rekomendasi Penanganan DBD",
    page_icon="",
    layout="wide"
)

# ============================================================
# KONSTANTA
# ============================================================

LABEL_COLS = ['gejala_berat', 'ada_nyamuk', 'risiko_lingkungan']

URUTAN_BULAN = ['Jan','Feb','Mar','Apr','May','Jun',
                'Jul','Aug','Sep','Oct','Nov','Dec']

WARNA_PENANGANAN = {
    'PSN / 3M Plus'               : '#4f86c6',
    'Fogging'                     : '#f59e0b',
    'Pemantauan Penanganan Medis' : '#ef4444',
}

# Setiap skenario kini disimpan sebagai SATU pipeline (TF-IDF + CART),
# hasil dari notebook training yang sudah tidak lagi memakai TruncatedSVD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILES = {
    'Semua Data'    : os.path.join(BASE_DIR, 'pipeline_semua.pkl'),
    'Musim Kemarau' : os.path.join(BASE_DIR, 'pipeline_kemarau.pkl'),
    'Musim Hujan'   : os.path.join(BASE_DIR, 'pipeline_hujan.pkl'),
}

GEOJSON_PATH = os.path.join(BASE_DIR, 'jawa.geojson')

NORMALISASI_PETA = {
    **NORMALISASI_PROVINSI,
    'DIY'                          : 'DI Yogyakarta',
    'JOGJA'                        : 'DI Yogyakarta',
    'JOGJAKARTA'                   : 'DI Yogyakarta',
    'D.I. YOGYAKARTA'              : 'DI Yogyakarta',
    'D.I YOGYAKARTA'               : 'DI Yogyakarta',
    'DAERAH ISTIMEWA YOGYAKARTA'   : 'DI Yogyakarta',
    'SPECIAL REGION OF YOGYAKARTA' : 'DI Yogyakarta',
}

METRIK_KOL_MAP = {
    "Prioritas Penanganan" : "__prioritas__",
    "Total Tweet"          : None,
    "Gejala Berat"         : "gejala_berat",
    "Ada Nyamuk"           : "ada_nyamuk",
    "Risiko Lingkungan"    : "risiko_lingkungan",
}

CHOROPLETH_COLORS = {
    "Total Tweet"      : "Blues",
    "Gejala Berat"     : "Reds",
    "Ada Nyamuk"       : "YlOrBr",
    "Risiko Lingkungan": "Greens",
}

BIRU = "#3b82f6"

# ============================================================
# CSS GLOBAL
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stMarkdown, button, input, select, textarea {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.viz-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 18px 20px 14px 20px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    margin-bottom: 16px;
    border-top: 3.5px solid #3b82f6;
}

.kpi-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 15px 16px;
    display: flex;
    align-items: center;
    gap: 13px;
    box-shadow: 0 2px 8px rgba(203,213,225,0.60), 0 1px 3px rgba(203,213,225,0.40);
    transition: box-shadow .18s, transform .18s;
    height: 80px;
    margin-bottom: 14px;
    border-top: 3.5px solid #3b82f6;
}
.kpi-card:hover {
    box-shadow: 0 6px 20px rgba(203,213,225,0.80), 0 2px 6px rgba(203,213,225,0.50);
    transform: translateY(-2px);
}
.kpi-icon {
    width: 40px; height: 40px; border-radius: 10px;
    background: #eff6ff;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.kpi-body { flex: 1; min-width: 0; }
.kpi-label {
    font-size: 11px; color: #94a3b8; font-weight: 600;
    text-transform: uppercase; letter-spacing: .5px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.kpi-value { font-size: 22px; font-weight: 800; line-height: 1.2; color: #3b82f6; }

.ph-body {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; gap: 8px;
    padding: 36px 10px; text-align: center;
    background: #f8fafc; border: 2px dashed #cbd5e1;
    border-radius: 12px; min-height: 180px;
}
.ph-title { font-size: 13px; font-weight: 600; color: #64748b; margin: 0; }
.ph-sub   { font-size: 12px; color: #94a3b8; margin: 0; }

.info-box    { background:#eff6ff; border-left:4px solid #3b82f6; padding:10px 14px; border-radius:10px; font-size:13px; margin-bottom:12px; }
.warning-box { background:#fffbeb; border-left:4px solid #f59e0b; padding:10px 14px; border-radius:10px; font-size:13px; margin-bottom:12px; }
.error-box   { background:#fef2f2; border-left:4px solid #ef4444; padding:10px 14px; border-radius:10px; font-size:13px; margin-bottom:12px; }
.success-box { background:#f0fdf4; border-left:4px solid #22c55e; padding:10px 14px; border-radius:10px; font-size:13px; margin-bottom:12px; }

.header-scorecard {
    background: rgba(219,234,254,0.45);
    border: 1.5px solid #bfdbfe;
    border-radius: 14px;
    padding: 16px 22px;
    margin-bottom: 16px;
}
.header-title { font-size: 22px; font-weight: 800; color: #1e40af; margin: 0 0 4px 0; }
.header-sub   { font-size: 12px; color: #3b82f6; margin: 0; }

.sec-title {
    font-size: 13px; font-weight: 700; color: #0f172a;
    display: flex; align-items: center; gap: 7px; margin-bottom: 10px;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
    overflow-y: auto;
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"]::-webkit-scrollbar { width: 4px; }
section[data-testid="stSidebar"]::-webkit-scrollbar-thumb { background:#334155; border-radius:4px; }

section[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 2px !important; display: flex !important; flex-direction: column !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    display: flex !important; align-items: center !important;
    padding: 10px 14px !important; border-radius: 8px !important;
    font-size: 13px !important; font-weight: 500 !important;
    color: #94a3b8 !important; cursor: pointer !important;
    transition: all .15s !important; width: 100% !important;
    box-sizing: border-box !important; margin: 0 !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(59,130,246,.18) !important; color: #93c5fd !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child { display: none !important; }
section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"],
section[data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] {
    background: linear-gradient(90deg, #1d4ed8, #3b82f6) !important;
    color: #ffffff !important; border-radius: 8px !important;
    box-shadow: 0 3px 10px rgba(59,130,246,.35) !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] * { color: #ffffff !important; }

section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: #1e3a5f !important; border: 1.5px dashed #60a5fa !important;
    border-radius: 8px !important; padding: 6px 8px !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background: #2563eb !important; color: #fff !important;
    border-radius: 6px !important; font-weight: 700 !important; border: none !important;
}
section[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(90deg, #1d4ed8, #3b82f6) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    font-weight: 700 !important; width: 100% !important; margin-top: 4px !important;
    box-shadow: 0 2px 8px rgba(59,130,246,.3) !important;
}
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background-color: #1e293b !important; border: 1px solid #334155 !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] div[data-baseweb="select"] * {
    color: #f1f5f9 !important;
}

div[data-baseweb="popover"] [role="listbox"],
ul[data-testid="stSelectboxVirtualDropdown"] {
    background-color: #ffffff !important;
}
div[data-baseweb="popover"] [role="listbox"] li,
div[data-baseweb="popover"] [role="listbox"] li *,
ul[data-testid="stSelectboxVirtualDropdown"] li,
ul[data-testid="stSelectboxVirtualDropdown"] li * {
    color: #0f172a !important;
}
div[data-baseweb="popover"] [role="listbox"] li:hover,
ul[data-testid="stSelectboxVirtualDropdown"] li:hover {
    background-color: #eff6ff !important;
}
div[data-baseweb="popover"] [role="listbox"] li[aria-selected="true"],
ul[data-testid="stSelectboxVirtualDropdown"] li[aria-selected="true"] {
    background-color: #dbeafe !important;
}

.stButton > button { border-radius: 8px; font-weight: 600; }
div[data-testid="stDataFrame"] { border-radius: 10px; }
.stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 13px; }
.stDownloadButton > button {
    width: 100% !important; font-size: 12px !important;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# IKON SVG
# ============================================================
BIRU_SVG = "#3b82f6"

SVG = {
    'tweet'   : f'<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{BIRU_SVG}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M23 3a10.9 10.9 0 01-3.14 1.53 4.48 4.48 0 00-7.86 3v1A10.66 10.66 0 013 4s-4 9 5 13a11.64 11.64 0 01-7 2c9 5 20 0 20-11.5a4.5 4.5 0 00-.08-.83A7.72 7.72 0 0023 3z"/></svg>',
    'fever'   : f'<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{BIRU_SVG}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 14.76V3.5a2.5 2.5 0 00-5 0v11.26a4.5 4.5 0 105 0z"/></svg>',
    'mosquito': f'<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{BIRU_SVG}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><line x1="12" y1="2" x2="12" y2="6"/><line x1="12" y1="18" x2="12" y2="22"/><line x1="4.93" y1="4.93" x2="7.76" y2="7.76"/><line x1="16.24" y1="16.24" x2="19.07" y2="19.07"/><line x1="2" y1="12" x2="6" y2="12"/><line x1="18" y1="12" x2="22" y2="12"/></svg>',
    'leaf'    : f'<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{BIRU_SVG}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 8C8 10 5.9 16.17 3.82 19.34A1 1 0 004.63 21C8 19 12.26 18 17 17c2-6 1-9 1-9z"/><path d="M4 19c0-3 1.5-6.5 6-9"/></svg>',
    'calendar': '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="18" rx="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>',
    'pie'     : '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21.21 15.89A10 10 0 118 2.83"/><path d="M22 12A10 10 0 0012 2v10z"/></svg>',
    'tag'     : '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M20.59 13.41l-7.17 7.17a2 2 0 01-2.83 0L2 12V2h10l8.59 8.59a2 2 0 010 2.82z"/><line x1="7" y1="7" x2="7.01" y2="7"/></svg>',
    'bar'     : '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>',
    'map'     : '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"/><line x1="8" y1="2" x2="8" y2="18"/><line x1="16" y1="6" x2="16" y2="22"/></svg>',
    'table'   : '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/></svg>',
}


# ============================================================
# FUNGSI HELPER UI
# ============================================================

def render_header(title: str, subtitle: str):
    st.markdown(
        f"<div class='header-scorecard'>"
        f"<div class='header-title'>{title}</div>"
        f"<div class='header-sub'>{subtitle}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

def render_kpi(label: str, val: int, icon_svg: str):
    st.markdown(
        f"<div class='kpi-card'>"
        f"<div class='kpi-icon'>{icon_svg}</div>"
        f"<div class='kpi-body'>"
        f"<div class='kpi-label'>{label}</div>"
        f"<div class='kpi-value'>{val:,}</div>"
        f"</div></div>",
        unsafe_allow_html=True
    )

def open_card(title: str):
    st.markdown(
        f"<div class='viz-card'>"
        f"<div class='sec-title'>"
        f"<span style='width:8px;height:8px;border-radius:50%;"
        f"background:#3b82f6;display:inline-block;'></span>{title}</div>",
        unsafe_allow_html=True
    )

def close_card():
    st.markdown("</div>", unsafe_allow_html=True)

def render_placeholder(icon_svg: str, title: str,
                       sub: str = "Upload dan proses data untuk menampilkan visualisasi."):
    st.markdown(
        f"<div class='ph-body'>{icon_svg}"
        f"<div class='ph-title'>{title}</div>"
        f"<div class='ph-sub'>{sub}</div></div>",
        unsafe_allow_html=True
    )


# ============================================================
# FUNGSI HELPER GRAFIK MATPLOTLIB
# ============================================================
BG = '#ffffff'

def _setup_ax(ax):
    ax.set_facecolor(BG)
    ax.grid(axis='y', color='#f1f5f9', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[['top','right','left','bottom']].set_visible(False)
    ax.tick_params(labelsize=8, colors='#64748b')

def _render_chart(fig):
    plt.tight_layout(pad=1.0)
    st.pyplot(fig, use_container_width=True)
    plt.close()

def grafik_donut_rekomendasi(df, ax):
    if 'penanganan_utama' not in df.columns:
        return
    dist   = df['penanganan_utama'].value_counts()
    colors = [WARNA_PENANGANAN.get(k, '#94a3b8') for k in dist.index]
    _, _, autotexts = ax.pie(
        dist.values, labels=None, colors=colors,
        autopct='%1.1f%%', pctdistance=0.78,
        startangle=90, wedgeprops=dict(width=0.55, edgecolor='white', linewidth=2)
    )
    for at in autotexts:
        at.set_fontsize(8); at.set_fontweight('600'); at.set_color('#ffffff')
    ax.legend(dist.index, loc='lower center', bbox_to_anchor=(0.5, -0.12),
              frameon=False, fontsize=8)

def grafik_bar_label(df, ax):
    x   = np.arange(len(LABEL_COLS))
    n1s = [int(df[c].sum()) for c in LABEL_COLS if c in df.columns]
    n0s = [len(df) - v for v in n1s]
    b1  = ax.bar(x-0.2, n1s, 0.38, label='Terdeteksi (1)',       color='#3b82f6', edgecolor='none')
    b0  = ax.bar(x+0.2, n0s, 0.38, label='Tidak Terdeteksi (0)', color='#dbeafe', edgecolor='none')
    maks = max(n1s + n0s) if (n1s + n0s) else 1
    for b in list(b1) + list(b0):
        h = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, h+maks*0.02,
                str(int(h)), ha='center', fontsize=8, color='#475569', fontweight='600')
    ax.set_xticks(x)
    ax.set_xticklabels(['Gejala Berat','Ada Nyamuk','Risiko Lingkungan'], fontsize=9)
    ax.set_ylabel("Jumlah Tweet", fontsize=9, color='#64748b')
    ax.legend(frameon=False, fontsize=8)
    _setup_ax(ax)

def grafik_bar_rekomendasi(df, ax):
    if 'penanganan_utama' not in df.columns:
        return
    dist   = df['penanganan_utama'].value_counts()
    colors = [WARNA_PENANGANAN.get(k, '#94a3b8') for k in dist.index]
    bars   = ax.barh(dist.index, dist.values, color=colors, edgecolor='none', height=0.45)
    maks   = max(dist.values) if len(dist) > 0 else 1
    for bar, val in zip(bars, dist.values):
        pct = val / len(df) * 100
        ax.text(val + maks*0.015, bar.get_y() + bar.get_height()/2,
                f"{val:,}  ({pct:.1f}%)", va='center', fontsize=8.5,
                fontweight='600', color='#334155')
    ax.set_xlabel("Jumlah Tweet", fontsize=9, color='#64748b')
    _setup_ax(ax)


# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model(skenario):
    """Muat SATU pipeline (TF-IDF + CART) untuk skenario terkait.
    Sejak SVD dihapus, tidak ada lagi objek terpisah untuk tfidf/svd/cart."""
    return joblib.load(MODEL_FILES[skenario])


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style='padding:20px 4px 14px 4px; text-align:center;'>
        <div style='font-size:14px; font-weight:800; color:#f1f5f9; letter-spacing:.6px; text-transform:uppercase;'>PANEL KONTROL</div>
        <div style='font-size:10px; color:#60a5fa; margin-top:3px;'>TF-IDF · CART</div>
    </div>
    <hr style='border-color:#1e3a5f; margin:0 0 14px 0;'>
    """, unsafe_allow_html=True)

    try:
        pipeline_sem = load_model('Semua Data')
        pipeline_kem = load_model('Musim Kemarau')
        pipeline_huj = load_model('Musim Hujan')
        model_ok = True
    except Exception as e:
        model_ok = False
        st.exception(e)
        pipeline_sem = None
        pipeline_kem = None
        pipeline_huj = None
        st.markdown("""<div class='error-box' style='font-size:11px;'>
            Model tidak ditemukan. Periksa file pipeline_*.pkl di direktori.</div>""",
            unsafe_allow_html=True)

    st.markdown(
        "<div style='font-size:11px; font-weight:600; color:#64748b; "
        "text-transform:uppercase; letter-spacing:.4px; margin-bottom:6px;'>"
        "Upload Berkas Mentah</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=['csv','xlsx','xls','json'],
                                label_visibility="collapsed", key="uploader")

    if uploaded and model_ok:
        fname = uploaded.name
        if st.session_state.get('nama_file_raw') != fname:
            df_raw_tmp, err = baca_file_upload(uploaded)
            if err:
                st.error(err)
            else:
                st.session_state['df_raw']        = df_raw_tmp
                st.session_state['nama_file_raw'] = fname
                for k in ['df_semua','df_kemarau','df_hujan',
                          'df_orig_semua','df_orig_kemarau','df_orig_hujan',
                          'n_dihapus','nama_file_aktif','laporan_prep','undo_stack']:
                    st.session_state.pop(k, None)
                st.rerun()
    elif uploaded and not model_ok:
        st.warning("Pastikan file pipeline_*.pkl tersedia terlebih dahulu.")

    st.markdown("<hr style='border-color:#1e3a5f; margin:10px 0 10px 0;'>",
                unsafe_allow_html=True)

    raw_ada  = 'df_raw'   in st.session_state
    data_ada = 'df_semua' in st.session_state

    if model_ok:
        if raw_ada and not data_ada:
            fname_info = st.session_state.get('nama_file_raw', '-')
            st.markdown(
                f"<div style='font-size:11px; color:#fbbf24; margin:4px 0 6px 0;'>"
                f"Berkas Siap: <b>{fname_info}</b><br>"
                f"({len(st.session_state['df_raw']):,} baris)</div>",
                unsafe_allow_html=True)

        if st.button("Proses & Generate Data", type="primary",
                     use_container_width=True, disabled=(not raw_ada),
                     key="btn_generate_main"):
            df_raw_now = st.session_state.get('df_raw')
            if df_raw_now is not None:
                with st.spinner("Memproses semua skenario..."):
                    df_s, lap, err = pipeline_preprocessing(
                        df_raw_now, pipeline_sem)
                    if err:
                        st.error(err)
                    else:
                        df_k, _, _ = pipeline_preprocessing(df_raw_now, pipeline_kem)
                        df_h, _, _ = pipeline_preprocessing(df_raw_now, pipeline_huj)
                        st.session_state.update({
                            'df_semua'        : df_s,
                            'df_kemarau'      : df_k,
                            'df_hujan'        : df_h,
                            'df_orig_semua'   : df_s.copy(),
                            'df_orig_kemarau' : df_k.copy(),
                            'df_orig_hujan'   : df_h.copy(),
                            'laporan_prep'    : lap,
                            'n_dihapus'       : 0,
                            'nama_file_aktif' : st.session_state.get('nama_file_raw', '-'),
                            'undo_stack'      : [],
                        })
                        st.rerun()
    else:
        st.button("Proses & Generate Data", type="primary",
                  use_container_width=True, disabled=True)

    st.markdown("<hr style='border-color:#1e3a5f; margin:14px 0 10px 0;'>",
                unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:11px; font-weight:600; color:#64748b; "
        "text-transform:uppercase; letter-spacing:.4px; margin-bottom:6px;'>"
        "Skenario Musim</div>", unsafe_allow_html=True)
    musim_filter = st.selectbox("", ["Semua", "Kemarau", "Hujan"],
                                key="musim_sidebar", label_visibility="collapsed")

    if data_ada:
        n_hapus = st.session_state.get('n_dihapus', 0)
        n_baris = len(st.session_state['df_semua'])
        st.markdown(f"""
        <div style='background:#1e3a5f; border-radius:10px; padding:10px 12px;
                    margin-top:10px; border:1px solid #1d4ed8;'>
            <div style='font-size:10px; color:#60a5fa; text-transform:uppercase; margin-bottom:3px;'>Berkas Aktif</div>
            <div style='font-size:11px; font-weight:700; color:#e2e8f0; margin-bottom:4px;
                        overflow:hidden; text-overflow:ellipsis; white-space:nowrap;'>
                {st.session_state.get('nama_file_aktif','-')}
            </div>
            <div style='font-size:11px; color:#86efac;'>{n_baris:,} baris aktif</div>
            {'<div style="font-size:11px;color:#fca5a5;">Dihapus manual: ' + str(n_hapus) + '</div>' if n_hapus > 0 else ''}
        </div>""", unsafe_allow_html=True)

        if st.button("Reset ke Data Awal", use_container_width=True, key="btn_reset"):
            st.session_state['df_semua']   = st.session_state['df_orig_semua'].copy()
            st.session_state['df_kemarau'] = st.session_state['df_orig_kemarau'].copy()
            st.session_state['df_hujan']   = st.session_state['df_orig_hujan'].copy()
            st.session_state['n_dihapus']  = 0
            st.session_state['undo_stack'] = []
            st.rerun()

    st.markdown("<hr style='border-color:#1e3a5f; margin:14px 0 10px 0;'>",
                unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:11px; font-weight:600; color:#64748b; "
        "text-transform:uppercase; letter-spacing:.4px; margin-bottom:8px;'>"
        "Navigasi</div>", unsafe_allow_html=True)
    menu = st.radio("", ["  Dashboard Utama", "  Review & Editor Data"],
                    label_visibility="collapsed")
    st.markdown("<div style='padding-bottom:40px;'></div>", unsafe_allow_html=True)


# ============================================================
# STATE GLOBAL
# ============================================================
data_siap = 'df_semua' in st.session_state

_MUSIM_KEY      = {'Semua': 'df_semua',      'Kemarau': 'df_kemarau',      'Hujan': 'df_hujan'}
_MUSIM_ORIG_KEY = {'Semua': 'df_orig_semua', 'Kemarau': 'df_orig_kemarau', 'Hujan': 'df_orig_hujan'}

_df_skenario = st.session_state.get(_MUSIM_KEY[musim_filter])
if _df_skenario is not None and musim_filter != 'Semua' and 'musim' in _df_skenario.columns:
    df_aktif = _df_skenario[_df_skenario['musim'] == musim_filter]
else:
    df_aktif = _df_skenario

has_data = df_aktif is not None and len(df_aktif) > 0


# ============================================================
# HALAMAN 1: DASHBOARD UTAMA
# ============================================================
if menu == "  Dashboard Utama":

    render_header(
        "Dashboard Pendukung Keputusan Intervensi DBD",
        "Sistem Rekomendasi berbasis TF-IDF dan Decision Tree CART"
    )

    if has_data:
        lap = st.session_state.get('laporan_prep', {})
        st.markdown(f"""
        <div class='success-box'>
            Pipeline berhasil &mdash; File: <u>{st.session_state.get('nama_file_aktif','-')}</u>
            &nbsp;|&nbsp; {lap.get('n_bersih', len(df_aktif)):,} data bersih
            &nbsp;|&nbsp; {lap.get('n_kosong',0):,} kosong dihapus
            &nbsp;|&nbsp; {lap.get('n_duplikat',0):,} duplikat dibuang
            &nbsp;|&nbsp; Skenario: <b>{musim_filter}</b>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class='info-box'>Masukkan data mentah di panel kiri,
        lalu jalankan <b>Proses &amp; Generate Data</b>.</div>""", unsafe_allow_html=True)

    # ── KPI ──
    kpi_defs = [
        ("Total Tweet Laporan", "tweet",             SVG['tweet']),
        ("Sinyal Gejala Berat", "gejala_berat",      SVG['fever']),
        ("Sinyal Ada Nyamuk",   "ada_nyamuk",        SVG['mosquito']),
        ("Risiko Lingkungan",   "risiko_lingkungan", SVG['leaf']),
    ]
    cols_kpi = st.columns(4)
    for col, (label, key, icon) in zip(cols_kpi, kpi_defs):
        val = 0
        if has_data:
            val = len(df_aktif) if key == "tweet" else \
                  int(df_aktif[key].sum()) if key in df_aktif.columns else 0
        with col:
            render_kpi(label, val, icon)

    # ── Baris 1: Grafik Bulanan (Plotly interaktif) + Donut ──
    col_l, col_r = st.columns([3, 2])

    with col_l:
        open_card("Tren Sebaran Temporal Laporan Per Bulan")
        if has_data:
            # Agregasi per bulan
            df_bulan = (
                df_aktif
                .assign(bulan_urut=pd.Categorical(
                    df_aktif['bulan_nama'], categories=URUTAN_BULAN, ordered=True))
                .sort_values('bulan_urut')
                .groupby('bulan_nama', observed=True)
                .agg(
                    jumlah       =('bulan_nama',        'count'),
                    gejala_berat =('gejala_berat',      'sum'),
                    ada_nyamuk   =('ada_nyamuk',         'sum'),
                    risiko_lingk =('risiko_lingkungan',  'sum'),
                )
                .reindex(URUTAN_BULAN, fill_value=0)
                .reset_index()
            )

            kemarau_bulan = ['May','Jun','Jul','Aug','Sep','Oct']
            colors_bar = [
                '#3b82f6' if b in kemarau_bulan else '#bfdbfe'
                for b in df_bulan['bulan_nama']
            ]

            fig_bulan = go.Figure()

            fig_bulan.add_trace(go.Bar(
                x=df_bulan['bulan_nama'],
                y=df_bulan['jumlah'],
                marker_color=colors_bar,
                text=df_bulan['jumlah'].apply(lambda v: str(v) if v > 0 else ''),
                textposition='outside',
                textfont=dict(size=11, color='#475569'),
                customdata=df_bulan[['gejala_berat','ada_nyamuk','risiko_lingk']].values,
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Total Tweet           : %{y}<br>"
                    "Gejala Berat          : %{customdata[0]}<br>"
                    "Ada Nyamuk            : %{customdata[1]}<br>"
                    "Risiko Lingkungan     : %{customdata[2]}"
                    "<extra></extra>"
                ),
                hoverlabel=dict(
                    bgcolor='#1e293b',
                    font_color='#f1f5f9',
                    font_size=12,
                    bordercolor='#3b82f6',
                ),
                showlegend=False,
            ))

            # Dummy trace untuk legenda musim
            fig_bulan.add_trace(go.Bar(
                x=[None], y=[None],
                marker_color='#3b82f6',
                name='Kemarau (Mei–Okt)',
                showlegend=True,
            ))
            fig_bulan.add_trace(go.Bar(
                x=[None], y=[None],
                marker_color='#bfdbfe',
                name='Hujan (Nov–Apr)',
                showlegend=True,
            ))

            fig_bulan.update_layout(
                height=300,
                margin=dict(t=20, b=10, l=10, r=10),
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                showlegend=True,
                legend=dict(
                    orientation='h',
                    yanchor='bottom', y=1.02,
                    xanchor='right',  x=1,
                    font=dict(size=11, color='#64748b'),
                ),
                xaxis=dict(
                    tickfont=dict(size=11, color='#64748b'),
                    showgrid=False,
                    zeroline=False,
                ),
                yaxis=dict(
                    tickfont=dict(size=10, color='#94a3b8'),
                    gridcolor='#f1f5f9',
                    zeroline=False,
                    title=dict(text='Jumlah Tweet', font=dict(size=11, color='#64748b')),
                ),
                bargap=0.35,
            )

            st.plotly_chart(fig_bulan, use_container_width=True)
        else:
            render_placeholder(SVG['calendar'], "Tren Distribusi Bulanan")
        close_card()

    with col_r:
        open_card("Proporsi Keputusan Intervensi")
        if has_data:
            fig, ax = plt.subplots(figsize=(5, 3.2))
            fig.patch.set_facecolor(BG)
            grafik_donut_rekomendasi(df_aktif, ax)
            _render_chart(fig)
        else:
            render_placeholder(SVG['pie'], "Proporsi Kebijakan")
        close_card()

    # ── Baris 2: Bar Label + Bar Rekomendasi ──
    col2_l, col2_r = st.columns(2)

    with col2_l:
        open_card("Volume Deteksi Atribut Sinyal")
        if has_data:
            fig, ax = plt.subplots(figsize=(7, 3.2))
            fig.patch.set_facecolor(BG)
            grafik_bar_label(df_aktif, ax)
            _render_chart(fig)
        else:
            render_placeholder(SVG['tag'], "Sebaran Sinyal Atribut")
        close_card()

    with col2_r:
        open_card("Volume Prioritas Penanganan (CART)")
        if has_data:
            fig, ax = plt.subplots(figsize=(7, 3.2))
            fig.patch.set_facecolor(BG)
            grafik_bar_rekomendasi(df_aktif, ax)
            _render_chart(fig)
        else:
            render_placeholder(SVG['bar'], "Volume Kebijakan")
        close_card()

    # ── Baris 3: Peta Spasial ──
    open_card("Pemetaan Distribusi Kasus Spasial Jawa")

    if not has_data:
        render_placeholder(SVG['map'], "Peta Sebaran Kloropleth Spasial")
    elif not os.path.exists(GEOJSON_PATH):
        st.warning("Berkas 'jawa.geojson' tidak ditemukan pada folder utama.")
    else:
        metrik_peta = st.selectbox(
            "Parameter Tampilan Peta:",
            list(METRIK_KOL_MAP.keys()),
            key="metrik_peta_card"
        )

        try:
            import folium
            from streamlit_folium import st_folium

            with open(GEOJSON_PATH, encoding='utf-8') as f:
                geojson_data = json.load(f)

            sample_props = (geojson_data['features'][0]['properties']
                            if geojson_data.get('features') else {})
            key_prov = next(
                (k for k in ['state','name','NAME','provinsi','PROVINSI','Provinsi']
                 if k in sample_props),
                list(sample_props.keys())[0] if sample_props else 'name',
            )

            geojson_lookup = {
                str(f['properties'].get(key_prov, '')).upper().strip():
                str(f['properties'].get(key_prov, ''))
                for f in geojson_data.get('features', [])
            }

            def cocokkan_provinsi(val: str) -> str:
                norm      = NORMALISASI_PETA.get(str(val).upper().strip(), str(val).strip())
                upper_key = norm.upper().strip()
                return geojson_lookup.get(upper_key, norm)

            df_peta = df_aktif.copy()
            df_peta['provinsi_map'] = df_peta['lokasi_hasil_mining'].astype(str).apply(cocokkan_provinsi)

            m = folium.Map(location=[-7.5, 111.0], zoom_start=7, tiles='CartoDB positron')

            if metrik_peta == "Prioritas Penanganan":
                dom = (
                    df_peta
                    .groupby(['provinsi_map','penanganan_utama'])
                    .size().reset_index(name='n')
                    .sort_values('n', ascending=False)
                    .drop_duplicates('provinsi_map')
                    .set_index('provinsi_map')
                )

                def style_prioritas(feature):
                    nama  = str(feature['properties'].get(key_prov, ''))
                    warna = WARNA_PENANGANAN.get(dom.loc[nama, 'penanganan_utama'], '#dddddd') \
                            if nama in dom.index else '#dddddd'
                    return {'fillColor': warna, 'fillOpacity': 0.78, 'color': '#ffffff', 'weight': 1}

                folium.GeoJson(
                    geojson_data, style_function=style_prioritas,
                    tooltip=folium.GeoJsonTooltip(fields=[key_prov], aliases=['Provinsi:'])
                ).add_to(m)

                m.get_root().html.add_child(folium.Element("""
                <div style='position:fixed;bottom:30px;left:30px;z-index:1000;
                            background:white;padding:12px 16px;border-radius:10px;
                            box-shadow:0 4px 16px rgba(0,0,0,.12);font-size:12px;'>
                    <div style='font-weight:700;margin-bottom:8px;color:#0f172a;'>Prioritas Penanganan</div>
                    <div style='display:flex;align-items:center;gap:8px;margin-bottom:5px;'>
                        <span style='background:#4f86c6;width:14px;height:14px;border-radius:3px;display:inline-block;'></span>
                        <span style='color:#334155;'>PSN / 3M Plus</span></div>
                    <div style='display:flex;align-items:center;gap:8px;margin-bottom:5px;'>
                        <span style='background:#f59e0b;width:14px;height:14px;border-radius:3px;display:inline-block;'></span>
                        <span style='color:#334155;'>Fogging</span></div>
                    <div style='display:flex;align-items:center;gap:8px;margin-bottom:5px;'>
                        <span style='background:#ef4444;width:14px;height:14px;border-radius:3px;display:inline-block;'></span>
                        <span style='color:#334155;'>Pemantauan Medis</span></div>
                    <div style='display:flex;align-items:center;gap:8px;'>
                        <span style='background:#dddddd;width:14px;height:14px;border-radius:3px;display:inline-block;'></span>
                        <span style='color:#94a3b8;'>Tidak ada data</span></div>
                </div>"""))

            else:
                kol = METRIK_KOL_MAP[metrik_peta]

                if kol is None:
                    agg = df_peta.groupby('provinsi_map').size().reset_index(name='nilai')
                else:
                    df_peta[kol] = pd.to_numeric(df_peta[kol], errors='coerce').fillna(0).astype(int)
                    agg = df_peta.groupby('provinsi_map')[kol].sum().reset_index()
                    agg.columns = ['provinsi_map', 'nilai']

                agg = agg.rename(columns={'provinsi_map': 'provinsi'})

                folium.Choropleth(
                    geo_data=geojson_data,
                    data=agg,
                    columns=['provinsi', 'nilai'],
                    key_on=f'feature.properties.{key_prov}',
                    fill_color=CHOROPLETH_COLORS.get(metrik_peta, 'YlOrRd'),
                    fill_opacity=0.78,
                    line_opacity=0.3,
                    legend_name=f"Jumlah — {metrik_peta}",
                    nan_fill_color='#e2e8f0',
                ).add_to(m)

                folium.GeoJson(
                    geojson_data,
                    style_function=lambda x: {'fillOpacity': 0, 'weight': 0},
                    tooltip=folium.GeoJsonTooltip(fields=[key_prov], aliases=['Provinsi:'])
                ).add_to(m)

                with st.expander(f"Tabel sebaran {metrik_peta} per provinsi"):
                    df_tbl = agg.sort_values('nilai', ascending=False).reset_index(drop=True)
                    df_tbl.columns = ['Provinsi', metrik_peta]
                    st.dataframe(df_tbl, use_container_width=True, hide_index=True)
                    st.caption(f"Nama provinsi di GeoJSON (key: `{key_prov}`): "
                               f"{', '.join(sorted(geojson_lookup.values()))}")

            st_folium(m, width=None, height=420, use_container_width=True)

        except ImportError:
            st.warning("Jalankan: pip install folium streamlit-folium")
        except Exception as e:
            st.error(f"Gagal render peta: {e}")

    close_card()

    # ── Baris 4: Tabel Eksplorasi ──
    open_card("Eksplorasi Dokumen Hasil Klasifikasi")

    if not has_data:
        render_placeholder(SVG['table'], "Tabel Repositori Hasil Rekomendasi")
    else:
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            opt_rek = df_aktif['penanganan_utama'].dropna().unique().tolist() \
                      if 'penanganan_utama' in df_aktif.columns else []
            filter_rek = st.multiselect("Filter Kebijakan:", opt_rek, default=opt_rek, key="tbl_rek")
            filter_rek = filter_rek or opt_rek
        with col_f2:
            opt_prov = sorted(df_aktif['lokasi_hasil_mining'].dropna().unique().tolist()) \
                       if 'lokasi_hasil_mining' in df_aktif.columns else []
            filter_prov = st.multiselect("Filter Provinsi:", opt_prov, default=opt_prov, key="tbl_prov")
            filter_prov = filter_prov or opt_prov
        with col_f3:
            kata_cari = st.text_input("Cari Kata:", placeholder="Masukkan kata kunci...", key="tbl_cari")

        df_fil = df_aktif.copy()
        if 'penanganan_utama'    in df_fil.columns: df_fil = df_fil[df_fil['penanganan_utama'].isin(filter_rek)]
        if 'lokasi_hasil_mining' in df_fil.columns: df_fil = df_fil[df_fil['lokasi_hasil_mining'].isin(filter_prov)]
        if kata_cari.strip():
            df_fil = df_fil[df_fil['clean_text'].str.contains(kata_cari.lower(), na=False)]

        st.caption(f"Menampilkan **{len(df_fil):,}** dari **{len(df_aktif):,}** rekaman")

        cols_tabel = ['teks_asli','lokasi_hasil_mining','bulan_nama','musim',
                      'gejala_berat','ada_nyamuk','risiko_lingkungan','penanganan_utama']
        cols_ada = [c for c in cols_tabel if c in df_fil.columns]
        st.dataframe(df_fil[cols_ada].reset_index(drop=True),
                     use_container_width=True, height=340, hide_index=True)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                "Unduh Hasil (.CSV)",
                data=df_fil[cols_ada].to_csv(index=False).encode('utf-8-sig'),
                file_name="hasil_dbd.csv", mime="text/csv",
                use_container_width=True, key="dl_csv_main")
        with col_d2:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine='openpyxl') as w:
                df_fil[cols_ada].to_excel(w, index=False, sheet_name='Hasil')
            st.download_button(
                "Unduh Hasil (.XLSX)", data=buf.getvalue(),
                file_name="hasil_dbd.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True, key="dl_xlsx_main")

    close_card()


# ============================================================
# HALAMAN 2: REVIEW & EDITOR DATA
# ============================================================
elif menu == "  Review & Editor Data":

    render_header(
        "Panel Rekayasa & Validasi Konteks Data",
        f"Eliminasi, pencarian, dan regenerasi ulang model klasifikasi — Skenario aktif: {musim_filter}"
    )

    if not data_siap:
        st.markdown("""<div class='info-box'><b>Belum ada data aktif.</b>
        Selesaikan pemrosesan berkas pada panel kontrol kiri.</div>""",
        unsafe_allow_html=True)
        st.stop()

    df_sesi = df_aktif

    if df_sesi is None or len(df_sesi) == 0:
        st.warning("Data untuk skenario terpilih belum tersedia. "
                   "Ubah pilihan 'Skenario Musim' pada panel kontrol kiri.")
        st.stop()

    _df_orig = st.session_state.get(_MUSIM_ORIG_KEY[musim_filter], df_sesi)
    if musim_filter != 'Semua' and 'musim' in _df_orig.columns:
        _df_orig = _df_orig[_df_orig['musim'] == musim_filter]
    n_original = len(_df_orig)
    n_dihapus  = st.session_state.get('n_dihapus', 0)
    if 'undo_stack' not in st.session_state:
        st.session_state['undo_stack'] = []

    c1, c2, c3, c4 = st.columns(4)
    for col, (lbl, val) in zip([c1,c2,c3,c4], [
        ("Sisa Baris Aktif",   len(df_sesi)),
        ("Volume Awal",        n_original),
        ("Dihapus Manual",     n_dihapus),
        ("Slot Undo Tersedia", len(st.session_state['undo_stack'])),
    ]):
        with col:
            render_kpi(lbl, val, "")

    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)

    open_card("Operasi Manipulasi Data")
    tab1, tab2, tab3 = st.tabs([
        "Centang & Eliminasi Baris",
        "Pencarian Kata Kunci",
        "Regenerate Model"
    ])

    with tab1:
        st.caption("Centang baris yang ingin dieliminasi, lalu klik tombol hapus.")
        cols_t = ['teks_asli','lokasi_hasil_mining','bulan_nama','musim',
                  'gejala_berat','ada_nyamuk','risiko_lingkungan','penanganan_utama']
        cols_a = [c for c in cols_t if c in df_sesi.columns]
        df_ed  = df_sesi[cols_a].copy()
        df_ed.insert(0, 'Pilih?', False)

        edited    = st.data_editor(
            df_ed, use_container_width=True, height=380, hide_index=False,
            column_config={'Pilih?': st.column_config.CheckboxColumn(
                label='Eliminasi?', default=False, width='small')},
            disabled=cols_a, key='editor_centang',
        )
        idx_pilih = edited.index[edited['Pilih?']].tolist()
        n_pilih   = len(idx_pilih)

        col_h1, col_h2 = st.columns([2, 5])
        with col_h1:
            if st.button(
                f"Hapus {n_pilih} Baris" if n_pilih > 0 else "Hapus Baris Terpilih",
                type="primary", disabled=(n_pilih == 0), key="btn_hapus"
            ):
                st.session_state['undo_stack'].append({
                    'df_semua'  : st.session_state['df_semua'].copy(),
                    'df_kemarau': st.session_state['df_kemarau'].copy(),
                    'df_hujan'  : st.session_state['df_hujan'].copy(),
                    'n_dihapus' : n_dihapus,
                })
                for k in ['df_semua','df_kemarau','df_hujan']:
                    df_tmp = st.session_state[k]
                    valid  = [i for i in idx_pilih if i in df_tmp.index]
                    st.session_state[k] = df_tmp.drop(index=valid).reset_index(drop=True)
                st.session_state['n_dihapus'] = n_dihapus + n_pilih
                st.success(f"Berhasil mengeliminasi {n_pilih} baris.")
                st.rerun()

        with col_h2:
            n_undo = len(st.session_state['undo_stack'])
            if st.button(f"Kembalikan Aksi (Undo: {n_undo})",
                         disabled=(n_undo == 0), key="btn_undo"):
                snap = st.session_state['undo_stack'].pop()
                for k in ['df_semua','df_kemarau','df_hujan','n_dihapus']:
                    st.session_state[k] = snap[k]
                st.success("Undo berhasil.")
                st.rerun()

    with tab2:
        kata = st.text_input("Masukkan Kata Pengotor:",
                             placeholder="Contoh: promosi, iklan, obat",
                             key="kata_review")
        if kata:
            mask     = df_sesi['clean_text'].str.contains(kata.lower(), na=False)
            df_cocok = df_sesi[mask]
            if len(df_cocok) == 0:
                st.info("Tidak ditemukan rekaman dengan frasa tersebut.")
            else:
                st.caption(f"Ditemukan **{len(df_cocok):,}** baris:")
                st.dataframe(
                    df_cocok[['teks_asli','lokasi_hasil_mining','musim','penanganan_utama']],
                    use_container_width=True, height=220, hide_index=False)

                if st.button(f"Hapus {len(df_cocok):,} Baris Secara Massal",
                             type="primary", key="btn_hapus_kata"):
                    st.session_state['undo_stack'].append({
                        'df_semua'  : st.session_state['df_semua'].copy(),
                        'df_kemarau': st.session_state['df_kemarau'].copy(),
                        'df_hujan'  : st.session_state['df_hujan'].copy(),
                        'n_dihapus' : n_dihapus,
                    })
                    for k in ['df_semua','df_kemarau','df_hujan']:
                        df_tmp = st.session_state[k]
                        mask_k = df_tmp['clean_text'].str.contains(kata.lower(), na=False)
                        st.session_state[k] = df_tmp[~mask_k].reset_index(drop=True)
                    st.session_state['n_dihapus'] = n_dihapus + len(df_cocok)
                    st.success(f"{len(df_cocok)} baris dihapus.")
                    st.rerun()

    with tab3:
        st.markdown("""<div class='warning-box'>
            Setelah eliminasi data, tekan tombol di bawah agar sistem
            menghitung ulang rekomendasi CART untuk semua skenario.
        </div>""", unsafe_allow_html=True)

        if st.button("Jalankan Regenerate", type="primary", key="btn_regen_action"):
            with st.spinner("Menghitung ulang..."):
                st.session_state['df_semua']   = regenerate(st.session_state['df_semua'],   pipeline_sem)
                st.session_state['df_kemarau'] = regenerate(st.session_state['df_kemarau'], pipeline_kem)
                st.session_state['df_hujan']   = regenerate(st.session_state['df_hujan'],   pipeline_huj)
            st.success("Regenerate selesai untuk semua skenario.")
            st.rerun()

        st.markdown("---")

        cols_dl = [c for c in ['teks_asli','lokasi_hasil_mining','bulan_nama','musim',
                               'gejala_berat','ada_nyamuk','risiko_lingkungan',
                               'penanganan_utama'] if c in df_sesi.columns]
        slug = musim_filter.replace(' ', '_').lower()
        col_d1, col_d2 = st.columns(2)

        with col_d1:
            st.download_button(
                "Ekspor Berkas (.CSV)",
                data=df_sesi[cols_dl].to_csv(index=False).encode('utf-8-sig'),
                file_name=f"hasil_{slug}.csv", mime="text/csv",
                use_container_width=True, key="dl_csv_review")
        with col_d2:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine='openpyxl') as w:
                df_sesi[cols_dl].to_excel(w, index=False, sheet_name='Hasil')
            st.download_button(
                "Ekspor Berkas (.XLSX)", data=buf.getvalue(),
                file_name=f"hasil_{slug}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True, key="dl_xlsx_review")

    close_card()