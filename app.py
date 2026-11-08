import json
import html
import gradio as gr
import config
import spaces
 
from pathlib import Path
from src.predictor import PrediktorGDM, PrediktorTigaModel 
 
# ================================================================
# KONFIGURASI DAN PEMUATAN MODEL
# ================================================================
 
ROOT = Path(__file__).resolve().parent

ARTIFACT_DIR = ROOT / "artifacts"
METADATA_PATH = ARTIFACT_DIR / "metadata_fitur.json"

FILE_MODEL = [
    "model_baseline.pkl",
    "model_gwo.pkl",
    "model_ao.pkl"
]

for nama_file in FILE_MODEL:
    if not (ARTIFACT_DIR / nama_file).exists():
        raise FileNotFoundError(
            f"File {nama_file} tidak ditemukan."
        )

prediktor = PrediktorTigaModel(ARTIFACT_DIR)
 
if METADATA_PATH.exists():
    meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
else:
    meta = {}
 
 
# ================================================================
# KELOMPOK FITUR
# ================================================================
 
fitur = prediktor.features
 
kelompok = {}
sudah = set()
 
for nama_kelompok, daftar in config.KELOMPOK_FITUR.items():
    tersedia = [f for f in daftar if f in fitur]
 
    if tersedia:
        kelompok[nama_kelompok] = tersedia
        sudah.update(tersedia)
 
sisa = [f for f in fitur if f not in sudah]
 
if sisa:
    kelompok["Lainnya"] = sisa
 
# Urutan fitur mengikuti urutan komponen pada formulir
fitur_form = [
    nama_fitur
    for daftar in kelompok.values()
    for nama_fitur in daftar
]
 
 
# ================================================================
# KOMPONEN INPUT
# ================================================================
LABEL_FITUR = {
    "age": "Usia Ibu",
    "address": "Tempat Tinggal",
    "arsenic": "Paparan Arsenik",
    "DMf": "Riwayat Diabetes Melitus pada Ayah Kandung",
    "DMm": "Riwayat Diabetes Melitus pada Ibu Kandung",
    "DMp": "Riwayat Diabetes Melitus pada Pasangan",
    "DMsibling": "Riwayat Diabetes Melitus pada Saudara Kandung",
    "DMpf": "Riwayat Diabetes Melitus pada Keluarga dari Pihak Ayah",
    "DMmf": "Riwayat Diabetes Melitus pada Keluarga dari Pihak Ibu",
    "obesity": "Riwayat Obesitas Sebelum Kehamilan",
    "pcos": "Riwayat Polycystic Ovary Syndrome",
    "abortion": "Riwayat Keguguran",
    "iud": "Riwayat Penggunaan Alat Kontrasepsi dalam Rahim",
    "nnd": "Riwayat Kematian Bayi pada Periode Neonatal",
    "prom": "Riwayat Ketuban Pecah Dini",
    "macrosomia": "Riwayat Bayi Makrosomia atau Berat Lahir >4 kg",
    "baby3.5": "Riwayat Melahirkan Bayi dengan Berat >3,5 kg",
    "eclampsia": "Riwayat Eklampsia",
    "gdm": "Riwayat Diabetes Gestasional",
    "hypo": "Riwayat Hipotiroid",
    "hyper": "Riwayat Hipertiroid",
    "htn": "Riwayat Hipertensi",
    "polyH": "Riwayat Polihidramnion",
    "primi": "Kehamilan Pertama",
    "multiP": "Pernah Mengalami Kehamilan Lebih dari Satu Kali",
    "ht": "Tinggi Badan (cm)",
    "wt1": "Berat Badan Kunjungan Pertama (kg)",
    "fbs1": "Kadar Gula Darah Puasa",
    "h2.1": "Kadar Gula Darah 2 Jam"
}

def buat_komponen(nama):
    informasi = meta.get(nama, {})

    label = (
        LABEL_FITUR.get(nama)
        or informasi.get("label")
        or nama
    )

    pilihan = informasi.get("pilihan")

    if pilihan:
        choices = [
            (p["label"], str(p["nilai"]))
            for p in pilihan
        ]

        return gr.Dropdown(
            choices=choices,
            label=f"{label} ({nama})",
            value=None,
            allow_custom_value=False,
            elem_classes=["input-field"]
        )

    return gr.Textbox(
        label=f"{label} ({nama})",
        placeholder="Kosongkan jika tidak diketahui",
        elem_classes=["input-field"]
    )
 
 
# ================================================================
# TAMPILAN HASIL (HTML)
# ================================================================
 
STATUS_AWAL = """
<div class="result-card idle">
    <div class="result-icon">?</div>
    <div class="result-label">MENUNGGU INPUT</div>
    <h2>Belum Ada Prediksi</h2>
    <p>Lengkapi data pemeriksaan, lalu tekan tombol prediksi.</p>
</div>
"""
 
DETAIL_AWAL = """
<div class="prob-box">
    <div class="prob-head">
        <span>Probabilitas GDM</span>
        <strong>--</strong>
    </div>
    <div class="prob-track"><div class="prob-fill idle-fill" style="width:0%"></div></div>
</div>
<div class="metrics-grid">
    <div class="metric-card">
        <span class="metric-label">Probabilitas Tidak GDM</span>
        <strong>--</strong>
    </div>
    <div class="metric-card">
        <span class="metric-label">Body Mass Index</span>
        <strong>--</strong>
    </div>
    <div class="metric-card wide">
        <span class="metric-label">Model Digunakan</span>
        <strong>--</strong>
    </div>
</div>
"""
 
 
# ================================================================
# INDIKATOR LANGKAH (HALAMAN 1 / HALAMAN 2)
# ================================================================
 
def stepper(aktif):
    def langkah(no, teks):
        if no < aktif:
            kelas, isi = "step done", "✓"
        elif no == aktif:
            kelas, isi = "step active", str(no)
        else:
            kelas, isi = "step", str(no)
        return (
            f'<div class="{kelas}">'
            f'<span class="step-num">{isi}</span>{teks}</div>'
        )
 
    return (
        '<div class="stepper">'
        + langkah(1, "Input Data")
        + '<div class="step-line"></div>'
        + langkah(2, "Hasil Prediksi")
        + "</div>"
    )
 
 
# ================================================================
# FUNGSI PREDIKSI
# ================================================================


@spaces.GPU(duration=10)
def prediksi_gdm(*nilai):
    data = dict(zip(fitur_form, nilai))

    try:
        hasil_semua = prediktor.prediksi_semua(data)

        daftar_model = []

        for nama_model, hasil in hasil_semua.items():

            # Status prediksi
            if hasil["terindikasi_gdm"]:
                status = """
                <div class="result-card positive">
                    <div class="result-icon">!</div>
                    <h2>Terindikasi GDM</h2>
                    <p>
                        Model mendeteksi indikasi diabetes gestasional.
                    </p>
                </div>
                """
                fill_class = "fill-positive"

            else:
                status = """
                <div class="result-card negative">
                    <div class="result-icon">✓</div>
                    <h2>Tidak terindikasi GDM</h2>
                    <p>
                        Model tidak mendeteksi indikasi diabetes gestasional.
                    </p>
                </div>
                """
                fill_class = "fill-negative"

            # Probabilitas GDM
            prob_gdm = max(
                0.0,
                min(100.0, float(hasil["probabilitas_gdm"]))
            )

            # Gabungkan status dan detail dalam satu kartu model
            kartu_model = f"""
            <div class="model-result">

                <div class="model-title">
                    {html.escape(nama_model)}
                </div>

                {status}

                <div class="prob-box">
                    <div class="prob-head">
                        <span>Probabilitas GDM</span>
                        <strong>{hasil['probabilitas_gdm']:.2f}%</strong>
                    </div>

                    <div class="prob-track">
                        <div class="prob-fill {fill_class}"
                             style="width:{prob_gdm:.2f}%">
                        </div>
                    </div>
                </div>

                <div class="metrics-grid">

                    <div class="metric-card">
                        <span class="metric-label">
                            Probabilitas Tidak GDM
                        </span>
                        <strong>
                            {hasil['probabilitas_tidak_gdm']:.2f}%
                        </strong>
                    </div>

                    <div class="metric-card">
                        <span class="metric-label">
                            Body Mass Index
                        </span>
                        <strong>{hasil['bmi']:.2f}</strong>
                    </div>

                </div>

            </div>
            """

            daftar_model.append(kartu_model)

        # Semua model ditampilkan dalam satu komponen HTML
        status_html = "\n".join(daftar_model)

        jumlah_kosong = len(
            hasil_semua["Random Forest Baseline"]["fitur_diisi_median"]
        )

        if jumlah_kosong:
            catatan = (
                f"**{jumlah_kosong} fitur** kosong telah diisi "
                "menggunakan median data pelatihan."
            )
        else:
            catatan = "Seluruh fitur telah diisi."

        return (
            gr.update(visible=False),
            gr.update(visible=True),
            "",
            status_html,
            "",
            catatan
        )

    except ValueError as e:
        error = html.escape(str(e))

        return (
            gr.update(visible=True),
            gr.update(visible=False),
            f'<div class="error-card">{error}</div>',
            STATUS_AWAL,
            DETAIL_AWAL,
            ""
        )

 
 
# ================================================================
# DESAIN ANTARMUKA
# ================================================================
 
CSS = """
 
/* ---------- Dasar ---------- */
 
:root {
    --bg: #070c18;
    --card: rgba(19, 30, 49, 0.72);
    --card-solid: #111b2e;
    --border: rgba(120, 150, 200, 0.18);
    --text: #f1f5f9;
    --muted: #94a3b8;
    --accent: #6366f1;
    --accent-2: #06b6d4;
}
 
body,
.gradio-container {
    background:
        radial-gradient(900px 500px at 10% -10%, rgba(99, 102, 241, 0.22), transparent 60%),
        radial-gradient(800px 500px at 100% 0%, rgba(6, 182, 212, 0.16), transparent 60%),
        var(--bg) !important;
}
 
.gradio-container {
    max-width: 1440px !important;
    margin: auto !important;
    padding: 24px !important;
}
 
footer {
    display: none !important;
}
 
 
/* ---------- Header ---------- */
 
.header {
    text-align: center;
    padding: 34px 15px 38px;
    margin-bottom: 18px;
    border: 1px solid var(--border);
    border-radius: 22px;
    background: linear-gradient(135deg, rgba(99,102,241,0.14), rgba(6,182,212,0.08));
    backdrop-filter: blur(8px);
}
 
.header-tag {
    display: inline-block;
    padding: 7px 16px;
    border: 1px solid rgba(129, 140, 248, 0.4);
    border-radius: 30px;
    background: rgba(99, 102, 241, 0.15);
    color: #ffffff !important;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.6px;
    margin-bottom: 16px;
}
 
.header h1 {
    font-size: 38px;
    font-weight: 800;
    margin: 0 0 12px;
    background: linear-gradient(90deg, #e0e7ff, #67e8f9);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}
 
.header p {
    color: #ffffff !important;
    font-size: 14px;
    line-height: 1.75;
    max-width: 680px;
    width: 100%;
    margin: 0 auto !important;
    text-align: center !important;
    display: block;
}

.chips {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 18px;
}
 
.chip {
    padding: 7px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    color: #ffffff !important;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid var(--border);
    text-align: center;
}
 
 
/* ---------- Judul bagian ---------- */
 
.section-heading h3 {
    color: #ffffff !important;
    font-size: 18px !important;
    font-weight: 750 !important;
    margin-bottom: 5px !important;
}

.section-description p {
    color: #e2e8f0 !important;
    font-size: 13px !important;
    margin-bottom: 14px !important;
}
 
 
/* ---------- Kartu kelompok fitur ---------- */
 
.feature-card {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 18px !important;
    margin-bottom: 14px !important;
    overflow: visible !important;
    /* PENTING: jangan pakai backdrop-filter / transform / filter di sini.
       Properti itu membuat daftar dropdown (position: fixed) salah posisi
       dan muncul jauh di atas, tidak menempel pada input yang dipilih. */
    backdrop-filter: none !important;
    filter: none !important;
    transform: none !important;
    transition: border-color .2s ease;
}
 
.feature-card ul.options,
.feature-card [role="listbox"] {
    z-index: 9999 !important;
}
 
.feature-card:hover {
    border-color: rgba(129, 140, 248, 0.45) !important;
}
 
.group-title {
    display: flex;
    align-items: center;
    gap: 10px;
    color: #e0e7ff;
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 12px;
}
 
.group-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 800;
    color: #fff;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
}
 
 
/* ---------- Label dan input ---------- */
 
.input-field {
    min-width: 0 !important;
}
 
.input-field label span,
.input-field label {
    color: #ffffff !important;
    font-size: 12px !important;
    font-weight: 600 !important;
}
 
.input-field input,
.input-field textarea,
.input-field .wrap,
.input-field [role="listbox"] {
    border-radius: 10px !important;
}
 
.feature-card input,
.feature-card [role="combobox"] {
    min-height: 40px !important;
}
 
.feature-card .wrap {
    min-width: 0 !important;
}
 
.feature-card .form {
    margin-bottom: 5px !important;
}
 
 
/* ---------- Warna kotak input (abu-abu transparan) ---------- */
 
/* Ubah variabel tema Gradio khusus di dalam kartu fitur */
.feature-card {
    --input-background-fill: rgba(148, 163, 184, 0.10);
    --input-background-fill-hover: rgba(148, 163, 184, 0.14);
    --input-background-fill-focus: rgba(148, 163, 184, 0.16);
    --input-border-color: rgba(148, 163, 184, 0.25);
    --input-border-color-hover: rgba(148, 163, 184, 0.40);
    --input-border-color-focus: rgba(203, 213, 225, 0.60);
    --input-shadow-focus: 0 0 0 3px rgba(148, 163, 184, 0.15);
    --block-background-fill: transparent;
}
 
/* Latar blok pembungkus dibuat transparan */
.input-field {
    background: transparent !important;
    border: none !important;
}
 
/* Kotak input: hanya SATU lapis berwarna per field.
   Dropdown punya beberapa lapis bawaan (.wrap > .wrap-inner > .secondary-wrap > input);
   kalau semuanya diberi warna transparan, hasilnya menumpuk. */
.input-field .wrap,
.input-field textarea,
.input-field input[type="text"]:not([role="listbox"]),
.input-field input[type="number"] {
    background: rgba(148, 163, 184, 0.10) !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(148, 163, 184, 0.25) !important;
    box-shadow: none !important;
}
 
/* Lapis dalam dropdown dibuat polos (tanpa warna, border, bayangan) */
.input-field .wrap .wrap-inner,
.input-field .wrap .secondary-wrap,
.input-field .wrap input {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #e2e8f0 !important;
}
 
.input-field .wrap:hover,
.input-field textarea:hover,
.input-field input[type="text"]:not([role="listbox"]):hover {
    background: rgba(148, 163, 184, 0.14) !important;
    border-color: rgba(148, 163, 184, 0.40) !important;
}
 
.input-field .wrap:focus-within,
.input-field textarea:focus,
.input-field input[type="text"]:not([role="listbox"]):focus {
    background: rgba(148, 163, 184, 0.16) !important;
    border-color: rgba(203, 213, 225, 0.60) !important;
    box-shadow: 0 0 0 3px rgba(148, 163, 184, 0.15) !important;
}
 
.input-field input::placeholder,
.input-field textarea::placeholder {
    color: #64748b !important;
}
 
/* Daftar pilihan dropdown */
.feature-card ul.options {
    background: #1a2538 !important;
    border: 1px solid rgba(148, 163, 184, 0.30) !important;
    border-radius: 10px !important;
}
 
.feature-card ul.options li {
    color: #e2e8f0 !important;
}
 
.feature-card ul.options li:hover,
.feature-card ul.options li.active {
    background: rgba(148, 163, 184, 0.18) !important;
}
 
.feature-card ul.options li.selected {
    background: rgba(148, 163, 184, 0.28) !important;
}
 
 
/* ---------- Label input (tulisan "age (age)", dst.) ---------- */
 
/* Tema Soft memberi label latar biru lewat variabel ini */
.feature-card {
    --block-title-background-fill: transparent;
    --block-title-text-color: #cbd5e1;
    --block-title-padding: 0;
    --block-title-border-width: 0px;
    --block-title-border-color: transparent;
    --block-label-background-fill: transparent;
    --block-label-text-color: #cbd5e1;
    --block-label-border-width: 0px;
}
 
/* Pengaman jika versi Gradio tidak mengikuti variabel di atas */
.input-field label,
.input-field label > span,
.input-field span[data-testid="block-info"] {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    color: #cbd5e1 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
}
 
/* Latar pembungkus kartu dibuat transparan agar menyatu */
.feature-card .block,
.feature-card .form,
.feature-card .styler {
    background: transparent !important;
    border-color: transparent !important;
    box-shadow: none !important;
}
 
 
/* ---------- Tombol ---------- */
 
.predict-button {
    border: none !important;
    border-radius: 14px !important;
    min-height: 54px !important;
    font-size: 16px !important;
    font-weight: 750 !important;
    letter-spacing: .3px;
    color: #fff !important;
    background: linear-gradient(90deg, var(--accent), var(--accent-2)) !important;
    box-shadow: 0 10px 28px rgba(99, 102, 241, 0.35) !important;
    transition: transform .15s ease, box-shadow .15s ease !important;
    margin-top: 6px !important;
}
 
.predict-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 14px 34px rgba(6, 182, 212, 0.4) !important;
}
 
 
/* ---------- Panel hasil ---------- */
 
.result-panel {
    max-width: 780px;
    width: 100%;
    margin: 0 auto !important;
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 20px !important;
    padding: 22px !important;
    backdrop-filter: blur(8px);
}
 
.result-panel h3 {
    color: #e2e8f0 !important;
    font-weight: 750 !important;
}
 
.result-panel p {
    color: #ffffff;
    font-size: 12px;
}
 
.model-result {
    background: rgba(19, 30, 49, 0.65);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 16px;
    margin: 16px 0;
}

.model-title {
    color: #e0e7ff;
    font-size: 15px;
    font-weight: 800;
    margin-bottom: 12px;
}

.result-card {
    padding: 28px 18px;
    border-radius: 16px;
    text-align: center;
    margin: 8px 0 16px;
    animation: muncul .4s ease;
}
 
@keyframes muncul {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
 
.idle {
    background: rgba(148, 163, 184, 0.08);
    border: 1px dashed rgba(148, 163, 184, 0.35);
    color: #cbd5e1;
}
 
/* Status positif GDM */
.positive {
    background: linear-gradient(
        160deg,
        rgba(190, 30, 65, 0.65),
        rgba(120, 15, 40, 0.75)
    ) !important;
    border: 1px solid #fb7185 !important;
    color: #ffffff !important;
}

/* Status negatif GDM */
.negative {
    background: linear-gradient(
        160deg,
        rgba(10, 120, 80, 0.65),
        rgba(5, 70, 50, 0.75)
    ) !important;
    border: 1px solid #34d399 !important;
    color: #ffffff !important;
}

/* Judul dan deskripsi status */
.result-card.positive h2,
.result-card.positive p,
.result-card.negative h2,
.result-card.negative p {
    color: #ffffff !important;
    opacity: 1 !important;
}
 
.result-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 44px;
    height: 44px;
    border-radius: 50%;
    margin: 0 auto 12px;
    font-size: 22px;
    font-weight: 800;
    background: rgba(255, 255, 255, 0.12);
}
 
.result-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.6px;
    opacity: 0.8;
}
 
.result-card h2 {
    font-size: 24px;
    font-weight: 800;
    margin: 9px 0;
}
 
.result-card p {
    font-size: 12px;
    line-height: 1.6;
    margin: 0;
    opacity: 0.85;
}
 
 
/* ---------- Bar probabilitas ---------- */
 
.prob-box {
    background: rgba(8, 14, 28, 0.7);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 12px;
}
 
.prob-head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 10px;
}
 
.prob-head span {
    color: #e2e8f0;
    font-size: 12px;
}
 
.prob-head strong {
    color: var(--text);
    font-size: 24px;
    font-weight: 800;
}
 
.prob-track {
    height: 10px;
    border-radius: 10px;
    background: rgba(148, 163, 184, 0.15);
    overflow: hidden;
}
 
.prob-fill {
    height: 100%;
    border-radius: 10px;
    transition: width .8s ease;
}
 
.fill-positive {
    background: linear-gradient(90deg, #fb7185, #e11d48);
}
 
.fill-negative {
    background: linear-gradient(90deg, #34d399, #059669);
}
 
.idle-fill {
    background: #475569;
}
 
 
/* ---------- Kartu metrik ---------- */
 
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 11px;
    margin-bottom: 14px;
}
 
.metric-card {
    background: rgba(8, 14, 28, 0.7);
    border: 1px solid var(--border);
    border-radius: 13px;
    padding: 15px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    min-width: 0;
}
 
.metric-card.wide {
    grid-column: span 2;
}
 
.metric-label {
    color: #e2e8f0;
    font-size: 11px;
    line-height: 1.5;
}
 
.metric-card strong {
    color: #ffffff;
    font-size: 20px;
    font-weight: 750;
    overflow-wrap: anywhere;
}
 
 
/* ---------- Catatan dan disclaimer ---------- */
 
.note-card {
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(129, 140, 248, 0.3);
    border-radius: 12px;
    padding: 12px 14px;
    color: #c7d2fe;
    font-size: 12px;
    line-height: 1.7;
    margin-top: 6px;
}
 
.disclaimer {
    color: #8192a9;
    font-size: 11px;
    line-height: 1.7;
    margin-top: 14px;
    padding: 12px 3px 0;
    border-top: 1px solid var(--border);
}
 
.error-card {
    background: rgba(160, 40, 70, 0.25);
    border: 1px solid #a33b50;
    color: #fecdd3;
    border-radius: 12px;
    padding: 15px;
    font-size: 13px;
    margin: 8px 0 16px;
}
 
 
 
/* ---------- Indikator langkah ---------- */
 
.stepper {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 14px;
    margin: 4px 0 22px;
}
 
.step {
    display: flex;
    align-items: center;
    gap: 9px;
    color: #cbd5e1 !important;
    font-size: 13px;
    font-weight: 700;
    white-space: nowrap;
}
 
.step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    font-size: 13px;
    font-weight: 800;
    background: rgba(148, 163, 184, 0.15);
    border: 1px solid var(--border);
}
 
.step.active {
    color: #ffffff !important;
}
 
.step.active .step-num {
    color: #fff;
    border: none;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.2);
}
 
.step.done {
    color: #6ee7b7 !important;
}
 
.step.done .step-num {
    color: #052e22;
    border: none;
    background: #34d399;
}
 
.step-line {
    width: 64px;
    height: 2px;
    border-radius: 2px;
    background: var(--border);
}
 
 
/* ---------- Tombol sekunder & hapus ---------- */
 
.back-button,
.clear-button {
    border-radius: 12px !important;
    min-height: 48px !important;
    font-weight: 700 !important;
}
 
.back-button {
    background: rgba(148, 163, 184, 0.1) !important;
    border: 1px solid var(--border) !important;
    color: #e2e8f0 !important;
}
 
.clear-button {
    background: rgba(225, 29, 72, 0.12) !important;
    border: 1px solid rgba(251, 113, 133, 0.5) !important;
    color: #fecdd3 !important;
}
 
.clear-button:hover {
    background: rgba(225, 29, 72, 0.25) !important;
}
 
/* ---------- Responsif ---------- */
@media (max-width: 768px) {

    .gradio-container {
        padding: 12px !important;
    }

    .header {
        padding: 22px 10px;
    }

    .header h1 {
        font-size: 26px;
    }

    /* Halaman hasil prediksi */
    .result-panel {
        position: static;
        width: 100% !important;
        max-width: 100% !important;
        margin: 0 auto !important;
        padding: 14px !important;
        box-sizing: border-box !important;
    }

    .result-panel > * {
        min-width: 0 !important;
    }

    .model-result {
        width: 100% !important;
        box-sizing: border-box !important;
        margin: 16px auto !important;
    }

    /* Kartu metrik */
    .metric-card {
        padding: 13px;
    }

    .metric-card strong {
        font-size: 18px;
    }

    /* Indikator langkah */
    .stepper {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        width: 100%;
        margin: 8px 0 22px;
    }

    .step {
        font-size: 11px;
        gap: 6px;
    }

    .step-num {
        width: 26px;
        height: 26px;
        min-width: 26px;
    }

    .step-line {
        width: 25px;
        flex-shrink: 0;
    }

    /* Judul dan deskripsi */
    .section-heading h3 {
        font-size: 17px !important;
    }

    .section-description p {
        font-size: 12px !important;
        line-height: 1.6;
    }
}

/* =========================================
   FIX VISIBILITAS TEKS MOBILE
   ========================================= */

/* Judul kelompok fitur:
   Demografis dan lingkungan, riwayat medis, dll. */
.feature-card .group-title {
    color: #ffffff !important;
    font-weight: 750 !important;
}

/* Semua teks di halaman hasil prediksi */
.result-panel,
.result-panel h1,
.result-panel h2,
.result-panel h3,
.result-panel p,
.result-panel span,
.result-panel strong,
.result-panel b {
    color: #ffffff !important;
}

/* Judul masing-masing model */
.model-title {
    color: #ffffff !important;
}

/* Label probabilitas dan metrik */
.prob-head span,
.metric-label {
    color: #e2e8f0 !important;
}

/* Angka probabilitas dan BMI */
.prob-head strong,
.metric-card strong {
    color: #ffffff !important;
}

/* Catatan hasil prediksi */
.note-card {
    color: #ffffff !important;
}

/* Disclaimer */
.disclaimer {
    color: #e2e8f0 !important;
}

/* Tulisan status prediksi */
.result-card h2,
.result-card p {
    color: inherit !important;
}

/* Stepper halaman input dan hasil */
.step {
    color: #e2e8f0 !important;
}

.step.active {
    color: #ffffff !important;
}

.step.done {
    color: #6ee7b7 !important;
}

/* Pengaman tampilan HP */
@media (max-width: 768px) {

    .feature-card .group-title {
        font-size: 14px !important;
        color: #ffffff !important;
    }

    .result-panel {
        text-align: center;
    }

    .model-title {
        font-size: 15px !important;
    }

    .metric-label,
    .prob-head span {
        font-size: 12px !important;
    }

    .result-card h2 {
        font-size: 21px !important;
    }

    .result-card p {
        font-size: 13px !important;
    }

    .disclaimer,
    .note-card {
        font-size: 12px !important;
        line-height: 1.7 !important;
    }
}

/* Placeholder menyesuaikan lebar kotak input */
.input-field input::placeholder,
.input-field textarea::placeholder {
    color: #94a3b8 !important;
    font-size: 12px !important;
    opacity: 1 !important;
}

/* Mencegah teks placeholder terpotong */
.input-field input {
    min-width: 0 !important;
    width: 100% !important;
    box-sizing: border-box !important;
}

.input-field textarea {
    width: 100% !important;
    box-sizing: border-box !important;
    resize: vertical !important;
}

/* Khusus layar laptop dan HP */
.input-field .wrap {
    min-width: 0 !important;
    width: 100% !important;
    box-sizing: border-box !important;
}
"""
 
 
# ================================================================
# MEMBANGUN APLIKASI
# ================================================================
 
with gr.Blocks(
    title="Deteksi Diabetes Gestasional"
) as demo:
 
    # HEADER
 
    gr.HTML("""
    <div class="header">
        <div class="header-tag">
            MACHINE LEARNING · GESTATIONAL DIABETES
        </div>
 
        <h1>Deteksi Diabetes Gestasional</h1>
 
        <p>
            Sistem prediksi diabetes gestasional menggunakan
            algoritma Random Forest dengan optimasi
            Aquila Optimizer dan Grey Wolf Optimizer.
        </p>
 
        <div class="chips">
            <span class="chip">Random Forest</span>
            <span class="chip">Aquila Optimizer</span>
            <span class="chip">Grey Wolf Optimizer</span>
        </div>
    </div>
    """)
 
    # ============================================================
    # HALAMAN 1: INPUT DATA
    # ============================================================
 
    with gr.Column(visible=True) as halaman_input:
 
        gr.HTML(stepper(1))
 
        gr.Markdown(
            "### Data Pemeriksaan",
            elem_classes=["section-heading"]
        )
 
        gr.Markdown(
            "Masukkan data sesuai kondisi dan riwayat pemeriksaan.",
            elem_classes=["section-description"]
        )
 
        komponen = []
 
        for nomor, (nama_kelompok, daftar) in enumerate(kelompok.items(), start=1):
 
            with gr.Group(elem_classes=["feature-card"]):
 
                gr.HTML(
                    f'<div class="group-title">'
                    f'<span class="group-badge">{nomor}</span>'
                    f'{html.escape(nama_kelompok)}'
                    f'</div>'
                )
 
                if nama_kelompok == "Faktor risiko kehamilan dan klinis":
 
                    # Maksimal 3 fitur dalam satu baris
                    jumlah_kolom = 3
 
                    for i in range(0, len(daftar), jumlah_kolom):
 
                        with gr.Row(equal_height=True):
                            for nama_fitur in daftar[i:i + jumlah_kolom]:
 
                                with gr.Column(scale=1, min_width=180):
                                    komponen.append(
                                        buat_komponen(nama_fitur)
                                    )
 
                else:
                    with gr.Row(equal_height=True):
                        for nama_fitur in daftar:
                            with gr.Column(scale=1, min_width=180):
                                komponen.append(
                                    buat_komponen(nama_fitur)
                                )
 
        # Pesan error (jika input tidak valid) tampil di halaman input
        kotak_error = gr.HTML("")
 
        tombol = gr.Button(
            "Prediksi Sekarang",
            variant="primary",
            size="lg",
            elem_classes=["predict-button"]
        )
 
    # ============================================================
    # HALAMAN 2: HASIL PREDIKSI
    # ============================================================
 
    with gr.Column(visible=False, elem_classes=["result-panel"]) as halaman_hasil:
 
        gr.HTML(stepper(2))
 
        gr.Markdown("### Hasil Analisis")
 
        hasil_status = gr.HTML(STATUS_AWAL)
 
        hasil_detail = gr.HTML(DETAIL_AWAL)
 
        hasil_catatan = gr.Markdown(
            elem_classes=["note-card"]
        )
 
        gr.HTML("""
        <div class="disclaimer">
            <b>Catatan:</b> Hasil prediksi merupakan keluaran
            model machine learning untuk keperluan penelitian.
            Hasil ini bukan pengganti diagnosis tenaga medis.
        </div>
        """)
 
        with gr.Row():
            tombol_ubah = gr.Button(
                "← Ubah Data",
                elem_classes=["back-button"]
            )
            tombol_hapus = gr.Button(
                "Hapus Semua & Mulai Lagi",
                elem_classes=["clear-button"]
            )
 
    # ============================================================
    # AKSI TOMBOL
    # ============================================================
 
    NAIK = "() => window.scrollTo({top: 0, behavior: 'smooth'})"
 
    def ubah_data():
        # Kembali ke halaman input, isian tetap dipertahankan
        return gr.update(visible=True), gr.update(visible=False)
 
    def hapus_semua():
        # Kosongkan seluruh isian, reset hasil, kembali ke halaman input
        kosong = [gr.update(value=None) for _ in komponen]
        return kosong + [
            gr.update(visible=True),
            gr.update(visible=False),
            "",
            STATUS_AWAL,
            DETAIL_AWAL,
            ""
        ]
 
    tombol.click(
        fn=prediksi_gdm,
        inputs=komponen,
        outputs=[
            halaman_input,
            halaman_hasil,
            kotak_error,
            hasil_status,
            hasil_detail,
            hasil_catatan
        ]
    ).then(fn=None, inputs=None, outputs=None, js=NAIK)
 
    tombol_ubah.click(
        fn=ubah_data,
        inputs=None,
        outputs=[halaman_input, halaman_hasil]
    ).then(fn=None, inputs=None, outputs=None, js=NAIK)
 
    tombol_hapus.click(
        fn=hapus_semua,
        inputs=None,
        outputs=komponen + [
            halaman_input,
            halaman_hasil,
            kotak_error,
            hasil_status,
            hasil_detail,
            hasil_catatan
        ]
    ).then(fn=None, inputs=None, outputs=None, js=NAIK)
 
 
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="cyan",
            neutral_hue="slate"
        ),
        css=CSS
    )

class PrediktorTigaModel:
    def __init__(self, direktori):
        self.prediktor = {
            "Random Forest Baseline": PrediktorGDM.muat(
                direktori / "model_baseline.pkl"
            ),
            "Random Forest + GWO": PrediktorGDM.muat(
                direktori / "model_gwo.pkl"
            ),
            "Random Forest + AO": PrediktorGDM.muat(
                direktori / "model_ao.pkl"
            ),
        }

        self.features = self.prediktor[
            "Random Forest Baseline"
        ].features

    def prediksi_semua(self, data):
        hasil = {}

        for nama, model in self.prediktor.items():
            hasil[nama] = model.prediksi(data)

        return hasil