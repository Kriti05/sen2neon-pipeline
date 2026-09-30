"""
app.py — SEN2NEON 4× Super-Resolution Streamlit frontend

Design: Aurora Nebula (theme.py)
Run:
    streamlit run app.py
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import plotly.graph_objects as go
from PIL import Image

from theme import (
    apply_theme,
    style_fig,
    SIDEBAR_MUTED,
    LINE_A,
    LINE_B,
    BAR_BASE,
    BAR_BEST,
    BAR_WORST,
)
from predict import (
    build_model_from_checkpoint,
    load_norm_stats,
    predict_tile,
    load_geotiff,
    save_geotiff,
    to_rgb,
    to_false_color,
    compute_ndvi,
    ndvi_to_rgb,
)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

DATA_ROOT = Path("SEN2NEON_5GB")
CHECKPOINT = Path("runs/sen2neon/checkpoints/best.pt")
RESULTS_DIR = Path("evaluation_results/hybrid_best")

BAND_NAMES = [
    "B1", "B2", "B3", "B4", "B5", "B6",
    "B7", "B8", "B8A", "B9", "B11", "B12",
]
SCALE = 4

# ── Page config FIRST ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="SEN2NEON · 4× SR",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="auto",
)

# ── Theme IMMEDIATELY after page config ────────────────────────────────────
apply_theme()


# ═══════════════════════════════════════════════════════════════════════════
# CACHED RESOURCES
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading model...")
def get_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, name, cfg = build_model_from_checkpoint(CHECKPOINT, device)
    return model, name, cfg, device


@st.cache_resource(show_spinner=False)
def get_stats():
    return load_norm_stats(DATA_ROOT)


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════

def _init_state() -> None:
    defaults = {
        "input_tile": None,
        "input_meta": None,
        "input_name": None,
        "enhanced": None,
        "uncertainty": None,
        "inference_ms": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ═══════════════════════════════════════════════════════════════════════════
# UI HELPERS — using st.html() for reliable HTML rendering
# ═══════════════════════════════════════════════════════════════════════════

def _safe_html(html: str) -> None:
    """Render HTML reliably — tries st.html(), falls back to st.markdown."""
    try:
        st.html(html)
    except AttributeError:
        st.markdown(html, unsafe_allow_html=True)


def hero(title: str, subtitle: str, pills: list | None = None) -> None:
    pills = pills or []
    pill_html = "".join(
        f'<span class="pill{" green" if p[1] else ""}">{p[0]}</span>'
        for p in pills
    )
    html = (
        '<div class="hero">'
        f'<h1>{title}</h1>'
        f'<p>{subtitle}</p>'
        f'<div>{pill_html}</div>'
        '</div>'
    )
    _safe_html(html)


def section(title: str) -> None:
    html = (
        '<div class="section-header">'
        '<div class="bar"></div>'
        f'<h3>{title}</h3>'
        '</div>'
    )
    _safe_html(html)


def spacer(px: int = 10) -> None:
    st.markdown(
        f"<div style='height:{px}px'></div>",
        unsafe_allow_html=True,
    )


def preview(arr, mode: str):
    if mode == "True Color":
        return to_rgb(arr)
    if mode == "False Color (NIR)":
        return to_false_color(arr)
    return ndvi_to_rgb(compute_ndvi(arr))


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    _safe_html(
        '<div style="text-align:center;padding:10px 0 6px;">'
        '<div style="font-size:2.2rem;line-height:1;">🛰️</div>'
        '<h2 style="margin:8px 0 2px;font-size:1.15rem;'
        'letter-spacing:.12em;font-weight:800;">SEN2NEON</h2>'
        f'<p style="color:{SIDEBAR_MUTED} !important;font-size:0.72rem;'
        'margin:0;opacity:.85;letter-spacing:.08em;">'
        'ORBITAL IMAGING LAB</p>'
        '</div>'
    )
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "01  Enhance",
            "02  Spectral",
            "03  Uncertainty",
            "04  Validation",
            "05  Applications",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Engine telemetry**")

    try:
        model, model_name, model_cfg, device = get_model()
        n_params = sum(p.numel() for p in model.parameters())
        st.success(f"✓ `{model_name}` on `{device.type}`")
        st.caption(f"{n_params:,} parameters · 4× reconstruction")
    except Exception as exc:
        st.error(f"Model load failed:\n\n{exc}")
        st.stop()

    stats = get_stats()
    if stats is not None:
        st.caption("✓ Normalization stats loaded")
    else:
        st.caption("⚠ No norm stats — using /10000 fallback")

    st.divider()
    st.caption("SEN2NEON · 4× SR · Hybrid CNN-Transformer")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — ENHANCE
# ═══════════════════════════════════════════════════════════════════════════

if page.startswith("01"):
    hero(
        "Sharper spatial detail. Same spectral stack.",
        "A 12-band Sentinel-2 tile enters at 10 m/px and is reconstructed to "
        "2.5 m/px. Inspect the result as imagery, not as a decorative dashboard.",
        pills=[("4× UPSCALE", False), ("12 BANDS", True), ("SPECTRAL-SAFE", False)],
    )

    col_upload, col_sample = st.columns([2, 1], gap="small")

    with col_upload:
        section("Input staging")
        uploaded = st.file_uploader(
            "Upload a 12-band GeoTIFF",
            type=["tif", "tiff"],
            help="Bands must be in order B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12",
            label_visibility="collapsed",
        )
        st.caption("500 MB per file · TIF")

    with col_sample:
        section("Local sample")
        patches_dir = DATA_ROOT / "patches"
        sample_files: list[Path] = []
        if patches_dir.exists():
            sample_files = sorted(
                list(patches_dir.glob("*.npy")) + list(patches_dir.glob("*.tif"))
            )[:8]
        if sample_files:
            sample_choice = st.selectbox(
                "Sample patch",
                ["—"] + [s.name for s in sample_files],
                label_visibility="collapsed",
            )
        else:
            sample_choice = "—"
            st.caption("No sample patches found")

    # ── Load input ─────────────────────────────────────────────────────────
    if uploaded is not None:
        tmp = Path("_uploaded.tif")
        tmp.write_bytes(uploaded.read())
        try:
            arr, meta = load_geotiff(tmp)
            st.session_state.input_tile = arr
            st.session_state.input_meta = meta
            st.session_state.input_name = uploaded.name
            st.session_state.enhanced = None
            st.session_state.uncertainty = None
        except Exception as exc:
            st.error(f"Failed to read GeoTIFF: {exc}")
        finally:
            tmp.unlink(missing_ok=True)

    elif sample_choice != "—":
        sample_path = patches_dir / sample_choice
        try:
            if sample_path.suffix == ".npy":
                arr = np.load(sample_path).astype(np.float32)
                if arr.ndim == 3 and arr.shape[0] != 12 and arr.shape[-1] == 12:
                    arr = arr.transpose(2, 0, 1)
                meta = None
            else:
                arr, meta = load_geotiff(sample_path)
            st.session_state.input_tile = arr
            st.session_state.input_meta = meta
            st.session_state.input_name = sample_choice
            st.session_state.enhanced = None
            st.session_state.uncertainty = None
        except Exception as exc:
            st.error(f"Failed to load sample: {exc}")

    tile = st.session_state.input_tile
    if tile is None:
        st.info("Stage a GeoTIFF or select a local sample to unlock reconstruction.")
        st.stop()

    # ── Info row ───────────────────────────────────────────────────────────
    spacer(6)
    section(f"Loaded · {st.session_state.input_name}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Shape", f"{tile.shape[1]}×{tile.shape[2]}")
    c2.metric("Bands", tile.shape[0])
    c3.metric("Resolution", "10 m/px")
    c4.metric(
        "Patches needed",
        f"{max(1, (tile.shape[1] // 64) * (tile.shape[2] // 64))}",
    )

    view_mode = st.radio(
        "Preview mode",
        ["True Color", "False Color (NIR)", "NDVI"],
        horizontal=True,
    )
    prev_in = preview(tile, view_mode)

    spacer(6)
    st.image(
        prev_in,
        caption=f"Input · {view_mode} · 10 m/px",
        use_container_width=True,
    )
    spacer(6)

    run = st.button(
        "⚡ Enhance (4× super-resolution)",
        type="primary",
        use_container_width=True,
    )

    # ── Inference ──────────────────────────────────────────────────────────
    if run:
        progress = st.progress(0.0, text="Preparing...")

        def on_progress(done: int, total: int) -> None:
            frac = done / max(total, 1)
            progress.progress(frac, text=f"Processing patch {done}/{total}...")

        t0 = time.time()
        try:
            enhanced, uncertainty = predict_tile(
                model, tile, stats, device, progress_callback=on_progress
            )
            elapsed_ms = (time.time() - t0) * 1000
            st.session_state.enhanced = enhanced
            st.session_state.uncertainty = uncertainty
            st.session_state.inference_ms = elapsed_ms
            progress.progress(1.0, text=f"✓ Done in {elapsed_ms / 1000:.2f}s")
        except Exception as exc:
            progress.empty()
            st.error(f"Inference failed: {exc}")
            st.exception(exc)

    # ── Results ────────────────────────────────────────────────────────────
    if st.session_state.enhanced is not None:
        enhanced = st.session_state.enhanced
        spacer(10)
        section("Enhanced output")

        c1, c2, c3 = st.columns(3)
        c1.metric("Output shape", f"{enhanced.shape[1]}×{enhanced.shape[2]}")
        c2.metric("Resolution", "2.5 m/px")
        c3.metric("Inference time", f"{st.session_state.inference_ms / 1000:.2f} s")

        prev_out = preview(enhanced, view_mode)

        spacer(6)
        col_l, col_r = st.columns(2, gap="small")
        with col_l:
            st.image(prev_in, caption="Input · 10 m/px", use_container_width=True)
        with col_r:
            st.image(
                prev_out,
                caption="Enhanced · 2.5 m/px · 4×",
                use_container_width=True,
            )

        spacer(6)
        section("Download")

        dl1, dl2 = st.columns(2, gap="small")
        with dl1:
            if st.session_state.input_meta is not None:
                out_path = Path("_enhanced.tif")
                save_geotiff(
                    out_path, enhanced, st.session_state.input_meta, scale=SCALE
                )
                with open(out_path, "rb") as f:
                    st.download_button(
                        "⬇️ Enhanced GeoTIFF (2.5 m, georeferenced)",
                        data=f.read(),
                        file_name=f"enhanced_4x_{st.session_state.input_name}",
                        mime="image/tiff",
                        use_container_width=True,
                    )
                out_path.unlink(missing_ok=True)
            else:
                buf = io.BytesIO()
                np.save(buf, enhanced)
                st.download_button(
                    "⬇️ Enhanced array (.npy)",
                    data=buf.getvalue(),
                    file_name=f"enhanced_4x_{st.session_state.input_name}.npy",
                    mime="application/octet-stream",
                    use_container_width=True,
                )
        with dl2:
            buf = io.BytesIO()
            Image.fromarray(prev_out).save(buf, format="PNG")
            st.download_button(
                "⬇️ Preview PNG",
                data=buf.getvalue(),
                file_name=f"enhanced_preview_{st.session_state.input_name}.png",
                mime="image/png",
                use_container_width=True,
            )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — SPECTRAL
# ═══════════════════════════════════════════════════════════════════════════

elif page.startswith("02"):
    hero(
        "Spectral consistency",
        "Confirms the model preserves the spectral signature of the input. "
        "The output is sharper — not recolored.",
        pills=[("SAM < 3° = EXCELLENT", True), ("12 BANDS PRESERVED", False)],
    )

    if st.session_state.enhanced is None:
        st.info("Run inference on the **01 Enhance** page first.")
        st.stop()

    tile = st.session_state.input_tile
    enhanced = st.session_state.enhanced

    mean_in = tile.mean(axis=(1, 2))
    mean_out = enhanced.mean(axis=(1, 2))

    section("Mean reflectance per band")

    fig = go.Figure()
    fig.add_scatter(
        x=BAND_NAMES,
        y=mean_in,
        name="Input (10 m)",
        mode="lines+markers",
        line=dict(color=LINE_A, width=3),
        marker=dict(size=9, line=dict(color="#FFFFFF", width=2)),
    )
    fig.add_scatter(
        x=BAND_NAMES,
        y=mean_out,
        name="Enhanced (2.5 m)",
        mode="lines+markers",
        line=dict(color=LINE_B, width=3, dash="dash"),
        marker=dict(size=9, line=dict(color="#FFFFFF", width=2)),
    )
    style_fig(fig, 400)
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(title="Spectral band")
    fig.update_yaxes(title="Mean reflectance")
    st.plotly_chart(fig, use_container_width=True)

    a = mean_in / max(np.linalg.norm(mean_in), 1e-8)
    b = mean_out / max(np.linalg.norm(mean_out), 1e-8)
    sam_deg = float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1))))
    corr = float(np.corrcoef(mean_in, mean_out)[0, 1])

    spacer(6)
    section("Metrics")

    c1, c2, c3 = st.columns(3, gap="small")
    c1.metric(
        "SAM (mean-spectrum)",
        f"{sam_deg:.2f}°",
        help="Spectral angle between input and output mean spectra. <3° is excellent.",
    )
    c2.metric(
        "Spectral correlation",
        f"{corr:.4f}",
        help="Pearson r between input and output band means.",
    )
    c3.metric("Bands preserved", "12 / 12")

    section("Interpretation")

    if sam_deg < 3:
        st.success(
            f"✓ SAM {sam_deg:.2f}° — excellent spectral preservation. "
            "The model is not hallucinating new spectral content."
        )
    elif sam_deg < 6:
        st.warning(
            f"⚠ SAM {sam_deg:.2f}° — acceptable, but some spectral drift. "
            "Worth checking on a per-band basis."
        )
    else:
        st.error(f"✗ SAM {sam_deg:.2f}° — significant spectral distortion.")

    with st.expander("Per-band values"):
        df = pd.DataFrame({
            "Band": BAND_NAMES,
            "Input mean": mean_in,
            "Enhanced mean": mean_out,
            "Δ (out − in)": mean_out - mean_in,
            "Ratio (out / in)": mean_out / np.maximum(mean_in, 1e-8),
        })
        st.dataframe(
            df.style.format({
                "Input mean": "{:.5f}",
                "Enhanced mean": "{:.5f}",
                "Δ (out − in)": "{:+.5f}",
                "Ratio (out / in)": "{:.3f}",
            }),
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 — UNCERTAINTY
# ═══════════════════════════════════════════════════════════════════════════

elif page.startswith("03"):
    hero(
        "Uncertainty map",
        "Some reconstructed detail is inferred by the model, not directly "
        "observed. High-gradient regions are flagged for validation.",
        pills=[("MODEL-INFERRED DETAIL", False), ("VALIDATE BEFORE USE", True)],
    )

    if st.session_state.enhanced is None:
        st.info("Run inference on the **01 Enhance** page first.")
        st.stop()

    enhanced = st.session_state.enhanced
    uncertainty = st.session_state.uncertainty

    section("Spatial view")
    col_a, col_b = st.columns(2, gap="small")

    with col_a:
        st.image(
            to_rgb(enhanced),
            caption="Enhanced · 2.5 m",
            use_container_width=True,
        )

    with col_b:
        base = to_rgb(enhanced).astype(np.float32) / 255.0
        gray = base.mean(axis=-1)
        if uncertainty.shape != gray.shape:
            u = Image.fromarray((uncertainty * 255).astype(np.uint8))
            u = u.resize((gray.shape[1], gray.shape[0]), Image.BILINEAR)
            uncertainty_r = np.asarray(u).astype(np.float32) / 255.0
        else:
            uncertainty_r = uncertainty

        alpha = np.clip(uncertainty_r * 1.2, 0, 0.85)
        red = np.array([1.0, 0.15, 0.1])
        overlay = gray[..., None] * (1 - alpha[..., None]) + red * alpha[..., None]
        overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
        st.image(
            overlay,
            caption="🔴 High-uncertainty regions",
            use_container_width=True,
        )

    spacer(6)
    section("Summary stats")

    c1, c2, c3 = st.columns(3, gap="small")
    c1.metric("Mean uncertainty", f"{uncertainty.mean():.3f}")
    c2.metric("High-risk (>0.5)", f"{(uncertainty > 0.5).mean() * 100:.1f}%")
    c3.metric("Max uncertainty", f"{uncertainty.max():.3f}")

    st.info(
        "**Reading this map:** red regions are where the model added the most "
        "fine-scale detail. These are candidates for manual validation. "
        "Details here are **model-inferred**, not directly observed — they "
        "should be cross-checked against reference data before use in "
        "critical applications."
    )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4 — VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

elif page.startswith("04"):
    hero(
        "Validation against high-resolution reference",
        "Metrics on the held-out test set (NEON 2.5 m reference).",
        pills=[("NEON 2.5 m REFERENCE", True), ("TEST SET", False)],
    )

    summary_path = RESULTS_DIR / "summary.json"
    if not summary_path.exists():
        st.warning(
            f"No evaluation results at `{summary_path}`. Run:\n\n"
            "```\npython evaluation/evaluate.py --data-root SEN2NEON_5GB "
            "--checkpoint runs/sen2neon/checkpoints/best.pt --model hybrid\n```"
        )
        st.stop()

    summary = json.loads(summary_path.read_text())
    m = summary["metrics"]

    st.markdown(
        f"**Model:** `{summary.get('model', '—')}` · "
        f"**Test patches:** {summary.get('num_test_patches', '—')} · "
        f"**Scale:** {summary.get('scale', '—')}×"
    )

    section("Aggregate metrics")
    c1, c2, c3, c4, c5, c6 = st.columns(6, gap="small")
    c1.metric("PSNR", f"{m['psnr_db']:.2f} dB")
    c2.metric("SSIM", f"{m['ssim']:.4f}")
    c3.metric("SAM", f"{m['sam_degrees']:.2f}°")
    c4.metric("ERGAS", f"{m['ergas']:.2f}")
    c5.metric("RMSE", f"{m['rmse']:.5f}")
    c6.metric("Spec. Corr", f"{m['spectral_correlation']:.4f}")

    spacer(6)
    section("Per-band PSNR")

    per_band = summary["per_band"]
    bands_order = summary["bands"]
    psnr_vals = [per_band[b]["psnr_db"] for b in bands_order]
    best_idx = int(np.argmax(psnr_vals))
    worst_idx = int(np.argmin(psnr_vals))

    colors = [BAR_BASE] * len(bands_order)
    colors[best_idx] = BAR_BEST
    colors[worst_idx] = BAR_WORST

    fig = go.Figure(
        go.Bar(
            x=bands_order,
            y=psnr_vals,
            marker=dict(color=colors),
            text=[f"{v:.1f}" for v in psnr_vals],
            textposition="outside",
        )
    )
    style_fig(fig, 400)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="Band")
    fig.update_yaxes(
        title="PSNR (dB)",
        range=[min(psnr_vals) - 3, max(psnr_vals) + 3],
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"🏆 Best: **{bands_order[best_idx]}** ({max(psnr_vals):.2f} dB) · "
        f"⚠️ Worst: **{bands_order[worst_idx]}** ({min(psnr_vals):.2f} dB) · "
        f"Spread: **{max(psnr_vals) - min(psnr_vals):.2f} dB**"
    )

    with st.expander("Full per-band metrics"):
        df = pd.DataFrame([
            {
                "Band": b,
                "MAE": per_band[b]["mae"],
                "RMSE": per_band[b]["rmse"],
                "PSNR (dB)": per_band[b]["psnr_db"],
            }
            for b in bands_order
        ])
        st.dataframe(
            df.style.format({
                "MAE": "{:.5f}",
                "RMSE": "{:.5f}",
                "PSNR (dB)": "{:.2f}",
            }),
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5 — APPLICATIONS
# ═══════════════════════════════════════════════════════════════════════════

elif page.startswith("05"):
    hero(
        "Real-world applications",
        "Where 4× super-resolution adds analytical value.",
        pills=[("URBAN", False), ("AGRICULTURE", True), ("DISASTER", False)],
    )

    section("Domain use cases")
    c1, c2, c3 = st.columns(3, gap="small")

    with c1:
        st.markdown("### 🏙️ Urban analysis")
        st.markdown(
            "- Small buildings become distinguishable\n"
            "- Narrow roads resolvable\n"
            "- Rooftop inventories\n"
            "- Informal settlement mapping"
        )

    with c2:
        st.markdown("### 🌾 Crop monitoring")
        st.markdown(
            "- Field boundaries sharpened\n"
            "- Crop-row separation\n"
            "- Per-plot irrigation status\n"
            "- Early crop-stress detection"
        )

    with c3:
        st.markdown("### 🌊 Disaster assessment")
        st.markdown(
            "- Localized damage mapping\n"
            "- Flood-edge delineation\n"
            "- Debris-field identification\n"
            "- Infrastructure intactness"
        )

    spacer(10)
    section("Demo flow")
    st.markdown(
        "1. Go to **01 Enhance**\n"
        "2. Upload an urban / agricultural / disaster-zone tile\n"
        "3. Switch preview to **False Color (NIR)** — vegetation and water boundaries pop\n"
        "4. Compare input and output side-by-side — look for **newly resolved edges**\n"
        "5. Open **02 Spectral** to confirm no spectral drift\n"
        "6. Open **03 Uncertainty** to see which regions need validation"
    )

    st.warning(
        "**Scientific honesty:** super-resolution reconstructs plausible "
        "fine-scale detail. It does not create new observations. Always "
        "validate high-uncertainty regions against reference data before "
        "critical decisions."
    )