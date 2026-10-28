
import os
import re
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =====================================================
# 1. KONFIGURASI HALAMAN
# =====================================================
st.set_page_config(
    page_title="AI Analyst | Analisis Sentimen",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "model_final_sentimen_tokopedia.joblib"

# =====================================================
# 2. DESAIN DASHBOARD
# =====================================================
st.markdown("""
<style>
:root {
    --green-dark: #08291D;
    --green: #16803D;
    --green-light: #E4F6E9;
    --background: #F4F8F5;
    --text: #182820;
    --muted: #748078;
    --border: #E1EAE3;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(
        circle at 15% 10%,
        #25A653 0%,
        #147B42 42%,
        #0A4B30 100%
    );
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stMainBlockContainer"] {
    background: #F8FBF9;
    border-radius: 22px;
    border: 1px solid rgba(255,255,255,.4);
    padding: 2rem;
    margin-top: 1rem;
    margin-bottom: 1rem;
    max-width: 1500px;
    box-shadow: 0 20px 60px rgba(0,0,0,.18);
}

h1 {
    color: #17251D !important;
    font-weight: 800 !important;
}

h2, h3, h4 {
    color: #203A2B !important;
    font-weight: 700 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg, #08291D 0%, #0E3D2A 100%
    );
}

[data-testid="stSidebar"] * {
    color: #F0F8F2;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,.15);
}

/* Kartu statistik */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E1EAE3;
    border-radius: 16px;
    padding: 18px;
    min-height: 120px;
    box-shadow: 0 5px 16px rgba(18,55,33,.05);
}

[data-testid="stMetricLabel"] {
    color: #728077 !important;
    font-size: 14px !important;
}

[data-testid="stMetricValue"] {
    color: #183A28 !important;
    font-weight: 800 !important;
}

/* Header dashboard */
.hero-card {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
    padding: 24px;
    margin-bottom: 24px;
    background: white;
    border: 1px solid #E1EAE3;
    border-radius: 18px;
    box-shadow: 0 5px 18px rgba(18,55,33,.05);
}

.hero-card h2 {
    margin: 4px 0;
    color: #17251D !important;
}

.hero-card p {
    margin-bottom: 0;
    color: #748078 !important;
}

.eyebrow {
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.5px;
    color: #16803D;
}

.status-pill {
    white-space: nowrap;
    background: #E4F6E9;
    color: #147A38;
    border: 1px solid #C7EED2;
    padding: 9px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
}

.insight-card {
    background: #E5F5E9;
    border: 1px solid #D2EBDD;
    border-radius: 15px;
    padding: 18px;
    min-height: 120px;
}

.insight-card h4 {
    margin-top: 0;
}

.insight-card p {
    color: #42634D;
    font-size: 14px;
}

/* Upload dan tombol */
[data-testid="stFileUploader"] {
    background: white;
    border: 1px dashed #91B69B;
    border-radius: 14px;
    padding: 12px;
}

.stButton > button {
    background: #15803D;
    color: white;
    border: 0;
    border-radius: 9px;
    font-weight: 700;
}

.stButton > button:hover {
    background: #166534;
    color: white;
}

.stDownloadButton > button {
    background: #0F6B3C;
    color: white;
    border: 0;
    border-radius: 9px;
    font-weight: 700;
}

[data-testid="stDataFrame"] {
    border: 1px solid #E1EAE3;
    border-radius: 12px;
}

hr {
    border-color: #E1EAE3;
}

footer, #MainMenu {
    visibility: hidden;
}

@media (max-width: 900px) {
    [data-testid="stMainBlockContainer"] {
        padding: 1rem;
        margin: .25rem;
    }

    .hero-card {
        align-items: flex-start;
        flex-direction: column;
    }
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# 3. MEMUAT MODEL
# =====================================================
@st.cache_resource
def load_model(path, modified_time):
    return joblib.load(path)

if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model tidak ditemukan: {MODEL_PATH}. "
        "Letakkan file model di folder yang sama dengan app.py."
    )
    st.stop()

try:
    model = load_model(
        MODEL_PATH,
        os.path.getmtime(MODEL_PATH)
    )
except Exception as e:
    st.error(f"Model gagal dimuat: {e}")
    st.stop()

# =====================================================
# 4. FUNGSI PENGOLAHAN DATA
# =====================================================
def deteksi_kolom(df):
    kandidat = [
        "review_text", "reviewcontent", "review_content",
        "reviewtext", "review", "ulasan", "content",
        "comment", "komentar", "text", "review_body",
        "reviewbody"
    ]

    mapping = {
        str(k).strip().lower(): k
        for k in df.columns
    }

    for nama in kandidat:
        if nama in mapping:
            kolom = mapping[nama]
            if df[kolom].notna().any():
                return kolom

    kolom_teks = df.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    if not kolom_teks:
        return None

    def skor(kolom):
        nilai = df[kolom].dropna().astype(str).str.strip()
        nilai = nilai[nilai.str.len() > 0]
        panjang = nilai.str.len().median() if not nilai.empty else 0
        return len(nilai), panjang

    return max(kolom_teks, key=skor)


def preprocessing_teks(teks):
    # Samakan dengan preprocessing saat pelatihan model.
    teks = str(teks).lower()
    teks = re.sub(r"https?://\S+|www\.\S+", " ", teks)
    teks = re.sub(r"[^a-zA-Z\s]", " ", teks)
    teks = re.sub(r"\s+", " ", teks).strip()
    return teks


# =====================================================
# 5. KATEGORISASI KELUHAN BERBASIS ATURAN
# =====================================================
aturan_keluhan = {
    "Kerusakan atau tidak berfungsi": [
        "rusak", "tidak berfungsi", "ga berfungsi",
        "gak berfungsi", "nggak berfungsi",
        "tdk berfungsi", "mati total",
        "tidak bisa dipakai", "error"
    ],
    "Ketidaksesuaian produk atau spesifikasi": [
        "tidak sesuai", "warna tdk sesuai",
        "warna tidak sesuai", "nyatanya",
        "bukan original", "beda dari gambar",
        "salah barang", "kapasitas"
    ],
    "Pengiriman dan pesanan": [
        "pengiriman", "kurir", "telat kirim",
        "terlambat", "belum sampai",
        "pesanan dibatalkan", "salah kirim",
        "paket hilang"
    ],
    "Garansi dan layanan purnajual": [
        "garansi", "tukar barang", "retur",
        "refund", "klaim garansi"
    ],
    "Kualitas dan performa produk": [
        "jelek", "tidak bagus", "kurang bagus",
        "virus", "tidak awet", "performa",
        "bermasalah"
    ],
    "Keaslian produk": [
        "palsu", "tidak original",
        "tidak ori", "bukan ori"
    ],
    "Pelayanan atau tanggung jawab penjual": [
        "tidak dibalas", "menipu", "penipuan",
        "tipu pembeli", "tidak bertanggung jawab"
    ]
}


def identifikasi_keluhan(teks):
    teks = str(teks).lower()
    kategori = []

    for nama, kata_kunci in aturan_keluhan.items():
        if any(k in teks for k in kata_kunci):
            kategori.append(nama)

    return "; ".join(kategori) if kategori else "Perlu ditinjau"


def baca_csv(file_csv):
    try:
        file_csv.seek(0)
        return pd.read_csv(file_csv)
    except UnicodeDecodeError:
        file_csv.seek(0)
        return pd.read_csv(file_csv, encoding="latin-1")


# =====================================================
# 6. PROSES ANALISIS
# =====================================================
def jalankan_analisis(df, kolom_review):
    hasil = df.copy()

    valid = (
        hasil[kolom_review].notna()
        & hasil[kolom_review].astype(str).str.strip().ne("")
    )

    if not valid.any():
        raise ValueError("Tidak ada ulasan yang bisa diproses.")

    teks = (
        hasil.loc[valid, kolom_review]
        .astype(str)
        .apply(preprocessing_teks)
        .tolist()
    )

    prediksi = model.predict(teks)

    hasil["prediksi_sentimen"] = "tidak tersedia"
    hasil.loc[valid, "prediksi_sentimen"] = prediksi

    # Distribusi sentimen
    jumlah = (
        hasil.loc[valid, "prediksi_sentimen"]
        .astype(str).str.lower().value_counts()
    )

    urutan = ["negatif", "netral", "positif"]

    distribusi = pd.DataFrame({
        "Sentimen": urutan,
        "Jumlah Ulasan": [
            int(jumlah.get(k, 0)) for k in urutan
        ]
    })

    total = int(distribusi["Jumlah Ulasan"].sum())

    distribusi["Persentase (%)"] = (
        distribusi["Jumlah Ulasan"] / total * 100
    ).round(2) if total else 0

    # Identifikasi ulasan negatif
    mask_negatif = (
        hasil["prediksi_sentimen"]
        .astype(str).str.lower().eq("negatif")
    )

    keluhan = hasil.loc[mask_negatif].copy()
    keluhan["kategori_keluhan"] = (
        keluhan[kolom_review].apply(identifikasi_keluhan)
    )

    # Frekuensi kategori keluhan
    terdeteksi = keluhan[
        keluhan["kategori_keluhan"] != "Perlu ditinjau"
    ].copy()

    if not terdeteksi.empty:
        terdeteksi["kategori_keluhan"] = (
            terdeteksi["kategori_keluhan"].str.split("; ")
        )
        terdeteksi = terdeteksi.explode("kategori_keluhan")

        frekuensi = (
            terdeteksi["kategori_keluhan"]
            .value_counts()
            .rename_axis("Kategori Keluhan")
            .reset_index(name="Jumlah Ulasan")
        )

        frekuensi.insert(
            0, "Peringkat",
            range(1, len(frekuensi) + 1)
        )
    else:
        frekuensi = pd.DataFrame(
            columns=[
                "Peringkat",
                "Kategori Keluhan",
                "Jumlah Ulasan"
            ]
        )

    return {
        "hasil": hasil,
        "distribusi": distribusi,
        "keluhan": keluhan,
        "frekuensi": frekuensi,
        "kolom_review": kolom_review,
        "total_valid": total
    }


# =====================================================
# 7. SESSION STATE
# =====================================================
if "analisis" not in st.session_state:
    st.session_state.analisis = None


# =====================================================
# 8. SIDEBAR NAVIGASI
# =====================================================
with st.sidebar:
    st.markdown(
        "<div style='color:#7DE49C;font-size:12px;"
        "font-weight:800;letter-spacing:2px'>"
        "▥ AI ANALYTICS</div>",
        unsafe_allow_html=True
    )

    st.markdown("# AI Analyst")
    st.caption("Marketplace Intelligence")
    st.divider()

    halaman = st.radio(
        "NAVIGASI",
        [
            "Dashboard",
            "Analisis Sentimen",
            "Analisis Keluhan",
            "Ekspor Data"
        ],
        key="menu_navigasi",
        format_func=lambda x: {
            "Dashboard": "▦  Dashboard",
            "Analisis Sentimen": "◉  Analisis Sentimen",
            "Analisis Keluhan": "⚑  Analisis Keluhan",
            "Ekspor Data": "⇩  Ekspor Data"
        }.get(x, x)
    )

    st.divider()
    st.markdown("**METODE MODEL**")
    st.markdown("• TF-IDF")
    st.markdown("• Regresi Logistik")

    st.divider()
    st.markdown("**STATUS SISTEM**")
    st.markdown(
        "<span style='color:#8BE3A5'>"
        "● Model berhasil dimuat</span>",
        unsafe_allow_html=True
    )

    st.caption("Tugas Akhir · Versi lokal")


# =====================================================
# 9. HEADER UTAMA
# =====================================================
st.markdown("""
<div class="hero-card">
    <div>
        <div class="eyebrow">
            MARKETPLACE SENTIMENT INTELLIGENCE
        </div>
        <h2>AI Analyst</h2>
        <p>
            Analisis sentimen ulasan produk dan identifikasi
            prioritas keluhan pelanggan.
        </p>
    </div>
    <span class="status-pill">● Model siap digunakan</span>
</div>
""", unsafe_allow_html=True)


# =====================================================
# 10. HALAMAN DASHBOARD
# =====================================================
if halaman == "Dashboard":
    st.markdown("### Ringkasan Analisis")
    st.write(
        "Pantau distribusi sentimen dan prioritas keluhan "
        "dari ulasan marketplace."
    )

    analisis = st.session_state.analisis

    if analisis is None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total ulasan", "—")
        c2.metric("Sentimen positif", "—")
        c3.metric("Sentimen netral", "—")
        c4.metric("Sentimen negatif", "—")

        st.info(
            "Belum ada hasil analisis. Buka menu "
            "'Analisis Sentimen' untuk mengunggah CSV."
        )

    else:
        dist = analisis["distribusi"].set_index("Sentimen")
        total = analisis["total_valid"]

        n_neg = int(dist.loc["negatif", "Jumlah Ulasan"])
        n_net = int(dist.loc["netral", "Jumlah Ulasan"])
        n_pos = int(dist.loc["positif", "Jumlah Ulasan"])

        # KARTU STATISTIK
        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Total ulasan",
            f"{total:,}"
        )

        c2.metric(
            "Sentimen positif",
            f"{n_pos:,}",
            f"{n_pos / total * 100:.1f}%" if total else "0%"
        )

        c3.metric(
            "Sentimen netral",
            f"{n_net:,}",
            f"{n_net / total * 100:.1f}%" if total else "0%"
        )

        c4.metric(
            "Sentimen negatif",
            f"{n_neg:,}",
            f"{n_neg / total * 100:.1f}%" if total else "0%"
        )

        st.divider()

        # GRAFIK DAN RINGKASAN
        kiri, kanan = st.columns([1.1, 1])

        with kiri:
            st.markdown("#### Distribusi Sentimen")

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(
                analisis["distribusi"]["Sentimen"],
                analisis["distribusi"]["Jumlah Ulasan"],
                color=["#DC2626", "#D97706", "#16A34A"],
                width=0.62
            )

            ax.set_ylabel("Jumlah ulasan")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_facecolor("white")
            fig.patch.set_facecolor("white")
            fig.tight_layout()

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with kanan:
            st.markdown("#### Ringkasan Data")

            st.dataframe(
                analisis["distribusi"],
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                f"Kolom ulasan: {analisis['kolom_review']}"
            )

            st.caption(
                "Nilai merupakan hasil prediksi model, "
                "bukan label aktual."
            )

        st.markdown("#### Insight Singkat")

        dominan = analisis["distribusi"].sort_values(
            "Jumlah Ulasan", ascending=False
        ).iloc[0]

        total_keluhan = len(analisis["keluhan"])

        i1, i2 = st.columns(2)

        with i1:
            st.markdown(
                f"""
                <div class="insight-card">
                    <h4>Sentimen Dominan</h4>
                    <p>
                    <b>{dominan['Sentimen'].title()}</b> merupakan
                    kelas dengan jumlah tertinggi, yaitu
                    {int(dominan['Jumlah Ulasan']):,} ulasan
                    ({dominan['Persentase (%)']:.1f}%).
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with i2:
            st.markdown(
                f"""
                <div class="insight-card">
                    <h4>Ulasan Negatif</h4>
                    <p>
                    Terdapat <b>{total_keluhan:,}</b> ulasan
                    yang diprediksi negatif. Buka menu Analisis
                    Keluhan untuk melihat kategorinya.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )


# =====================================================
# 11. HALAMAN ANALISIS SENTIMEN
# =====================================================
elif halaman == "Analisis Sentimen":
    st.subheader("Unggah Dataset dan Prediksi Sentimen")

    st.write(
        "Unggah file CSV berisi ulasan produk marketplace."
    )

    file_csv = st.file_uploader(
        "Pilih dataset ulasan (CSV)",
        type=["csv"],
        key="upload_csv"
    )

    if file_csv is not None:
        try:
            df = baca_csv(file_csv)
        except Exception as e:
            st.error(f"CSV gagal dibaca: {e}")
            st.stop()

        if df.empty:
            st.error("Dataset kosong.")
            st.stop()

        st.caption(
            f"Ukuran dataset: {df.shape[0]:,} baris × "
            f"{df.shape[1]} kolom"
        )

        kolom_awal = deteksi_kolom(df)

        if kolom_awal is None:
            st.error(
                "Kolom teks tidak terdeteksi. "
                "Periksa kembali dataset."
            )
            st.stop()

        kolom_review = st.selectbox(
            "Kolom teks ulasan (terdeteksi otomatis)",
            options=df.columns.tolist(),
            index=df.columns.tolist().index(kolom_awal)
        )

        with st.expander("Pratinjau dataset"):
            st.dataframe(
                df.head(10),
                use_container_width=True,
                hide_index=True
            )

        if st.button(
            "▶ Mulai Analisis",
            type="primary",
            use_container_width=True
        ):
            with st.spinner(
                "Memprediksi sentimen dan menganalisis keluhan..."
            ):
                try:
                    st.session_state.analisis = jalankan_analisis(
                        df, kolom_review
                    )
                    st.success(
                        "Analisis berhasil! Buka menu Dashboard "
                        "untuk melihat ringkasannya."
                    )
                except Exception as e:
                    st.error(
                        "Analisis gagal. Periksa preprocessing "
                        "dan format model. Detail: " + str(e)
                    )

    if st.session_state.analisis is not None:
        st.divider()
        st.markdown("#### Pratinjau Hasil Prediksi")

        st.dataframe(
            st.session_state.analisis["hasil"].head(20),
            use_container_width=True,
            hide_index=True
        )


# =====================================================
# 12. HALAMAN ANALISIS KELUHAN
# =====================================================
elif halaman == "Analisis Keluhan":
    st.subheader("Analisis Kategori Keluhan Pelanggan")

    analisis = st.session_state.analisis

    if analisis is None:
        st.info(
            "Belum ada hasil analisis. Unggah CSV pada menu "
            "Analisis Sentimen terlebih dahulu."
        )

    else:
        keluhan = analisis["keluhan"]
        frekuensi = analisis["frekuensi"]

        ditinjau = int(
            keluhan["kategori_keluhan"]
            .eq("Perlu ditinjau").sum()
        ) if not keluhan.empty else 0

        c1, c2, c3 = st.columns(3)

        c1.metric("Total ulasan negatif", f"{len(keluhan):,}")
        c2.metric(
            "Teridentifikasi",
            f"{len(keluhan) - ditinjau:,}"
        )
        c3.metric("Perlu ditinjau", f"{ditinjau:,}")

        st.caption(
            "Kategori keluhan menggunakan aturan kata kunci, "
            "bukan prediksi kategori oleh Regresi Logistik. "
            "Satu ulasan dapat masuk ke beberapa kategori."
        )

        if frekuensi.empty:
            st.info(
                "Belum ada kategori keluhan yang cocok "
                "dengan aturan kata kunci."
            )

        else:
            kiri, kanan = st.columns([1, 1])

            with kiri:
                st.markdown("#### Frekuensi Kategori")

                st.dataframe(
                    frekuensi,
                    use_container_width=True,
                    hide_index=True
                )

            with kanan:
                st.markdown("#### Prioritas Keluhan")

                plot_data = frekuensi.sort_values(
                    "Jumlah Ulasan", ascending=True
                )

                fig, ax = plt.subplots(figsize=(8, 5))

                ax.barh(
                    plot_data["Kategori Keluhan"],
                    plot_data["Jumlah Ulasan"],
                    color="#16803D"
                )

                ax.set_xlabel("Jumlah kecocokan kategori")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        with st.expander("Lihat ulasan negatif"):
            st.dataframe(
                keluhan,
                use_container_width=True,
                hide_index=True
            )


# =====================================================
# 13. HALAMAN EKSPOR DATA
# =====================================================
elif halaman == "Ekspor Data":
    st.subheader("Ekspor Hasil Analisis")

    analisis = st.session_state.analisis

    if analisis is None:
        st.info(
            "Belum ada hasil yang bisa diunduh. "
            "Jalankan analisis terlebih dahulu."
        )

    else:
        hasil = analisis["hasil"]
        distribusi = analisis["distribusi"]
        keluhan = analisis["keluhan"]
        frekuensi = analisis["frekuensi"]

        st.write(
            "Pilih hasil analisis yang ingin disimpan "
            "ke komputer."
        )

        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                "⬇ Unduh Hasil Prediksi",
                data=hasil.to_csv(index=False).encode("utf-8-sig"),
                file_name="hasil_prediksi_sentimen.csv",
                mime="text/csv",
                use_container_width=True
            )

            st.download_button(
                "⬇ Unduh Distribusi Sentimen",
                data=distribusi.to_csv(index=False).encode("utf-8-sig"),
                file_name="distribusi_sentimen.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col2:
            st.download_button(
                "⬇ Unduh Kategorisasi Keluhan",
                data=keluhan.to_csv(index=False).encode("utf-8-sig"),
                file_name="hasil_kategorisasi_keluhan.csv",
                mime="text/csv",
                use_container_width=True
            )

            st.download_button(
                "⬇ Unduh Frekuensi Keluhan",
                data=frekuensi.to_csv(index=False).encode("utf-8-sig"),
                file_name="frekuensi_kategori_keluhan.csv",
                mime="text/csv",
                use_container_width=True
            )


# =====================================================
# 14. FOOTER
# =====================================================
st.divider()

st.caption(
    "AI Analyst | TF-IDF + Regresi Logistik | "
    "Analisis Sentimen Ulasan Marketplace"
)
