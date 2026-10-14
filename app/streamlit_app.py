"""Professional Streamlit interface for casting defect inspection and XAI."""
from __future__ import annotations

import base64
import io
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gradcam import make_gradcam, make_visuals  # noqa: E402
from src.model_loader import load_config, load_model  # noqa: E402
from src.predict import predict, preprocess_pil_image  # noqa: E402

st.set_page_config(
    page_title="Casting Defect Inspector",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _default_users() -> dict[str, str]:
    return {
        "admin": "admin123",
        "admin123": "admin123",
        "inspector": "factory2025",
    }


def get_demo_users() -> dict[str, str]:
    try:
        users = st.secrets.get("app", {}).get("users", {})
    except Exception:
        users = {}
    if users:
        return {str(key): str(value) for key, value in users.items()}
    return _default_users()


def initialize_session() -> None:
    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("username", "")
    st.session_state.setdefault("light_theme", False)


def history_database() -> Path:
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir / "inspection_history.sqlite3"


def initialize_history() -> None:
    with sqlite3.connect(history_database()) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                file_name TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                defect_probability REAL NOT NULL,
                ok_probability REAL NOT NULL,
                operator_note TEXT NOT NULL,
                username TEXT NOT NULL
            )
            """
        )


def load_history() -> list[dict[str, object]]:
    with sqlite3.connect(history_database()) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT timestamp, file_name, prediction, confidence,
                   defect_probability, ok_probability, operator_note, username
            FROM inspections
            ORDER BY id DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def save_inspection(entry: dict[str, object]) -> None:
    with sqlite3.connect(history_database()) as connection:
        connection.execute(
            """
            INSERT INTO inspections (
                timestamp, file_name, prediction, confidence,
                defect_probability, ok_probability, operator_note, username
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry["timestamp"],
                entry["file_name"],
                entry["prediction"],
                entry["confidence"],
                entry["defect_probability"],
                entry["ok_probability"],
                entry["operator_note"],
                entry["username"],
            ),
        )


def app_theme() -> None:
    background_path = (
        ROOT
        / "casting_512x512"
        / "casting_512x512"
        / "ok_front"
        / "cast_ok_0_1018.jpeg"
    )
    background_image = base64.b64encode(background_path.read_bytes()).decode("ascii")
    light_theme = st.session_state.light_theme
    palette = (
        {
            "text": "#172033",
            "muted": "#475569",
            "surface": "rgba(255,255,255,.88)",
            "input": "#ffffff",
            "border": "#cbd5e1",
            "overlay": "rgba(241,245,249,.91), rgba(226,232,240,.87)",
        }
        if light_theme
        else {
            "text": "#e2e8f0",
            "muted": "#cbd5e1",
            "surface": "rgba(15,23,42,.88)",
            "input": "#111e31",
            "border": "#475569",
            "overlay": "rgba(5,14,27,.82), rgba(8,21,39,.78)",
        }
    )
    st.markdown(
        """
        <style>
        .stApp {
            --app-text: __TEXT__;
            --app-muted: __MUTED__;
            --app-surface: __SURFACE__;
            --app-input: __INPUT__;
            --app-border: __BORDER__;
            background-image:
                linear-gradient(155deg, __OVERLAY__),
                url("data:image/jpeg;base64,__BACKGROUND_IMAGE__");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: var(--app-text) !important;
        }
        .block-container { max-width: 1420px; padding: 2rem 2rem 4rem; }
        header[data-testid="stHeader"] { background:transparent !important; }
        header[data-testid="stHeader"] button { color:var(--app-text) !important; }
        .stApp [data-testid="stSidebar"] { background-color: var(--app-surface); }
        .stApp [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: var(--app-surface);
            border-radius: 16px;
        }
        .stApp [data-testid="stMarkdownContainer"] * ,
        .stApp [data-testid="stWidgetLabel"],
        .stApp [data-testid="stWidgetLabel"] p,
        .stApp [data-testid="stMetricLabel"],
        .stApp [data-testid="stMetricValue"],
        .stApp [data-testid="stCaptionContainer"],
        .stApp [data-testid="stTabs"] button,
        .stApp [data-testid="stFileUploader"] small,
        .stApp [data-testid="stFileUploader"] label {
            color: var(--app-text) !important;
        }
        .stApp [data-testid="stMarkdownContainer"] .hero,
        .stApp [data-testid="stMarkdownContainer"] .hero *,
        .stApp [data-testid="stMarkdownContainer"] .login-panel,
        .stApp [data-testid="stMarkdownContainer"] .login-panel * {
            color:#f1f5f9 !important;
        }
        .stApp [data-testid="stMarkdownContainer"] .hero p { color:#dbeafe !important; }
        .stApp [data-testid="stMarkdownContainer"] .login-panel p { color:#cbd5e1 !important; }
        .stApp [data-testid="stMarkdownContainer"] .result-card,
        .stApp [data-testid="stMarkdownContainer"] .result-card * { color:#f1f5f9 !important; }
        .stApp [data-testid="stMarkdownContainer"] .result-card .ok { color:#34d399 !important; }
        .stApp [data-testid="stMarkdownContainer"] .result-card .defective { color:#fb7185 !important; }
        .stApp [data-testid="stCaptionContainer"] { color:var(--app-muted) !important; }
        .stApp input, .stApp textarea, .stApp [data-baseweb="select"] > div {
            background-color: var(--app-input);
            color: var(--app-text) !important;
            border-color: var(--app-border);
        }
        .stApp input::placeholder, .stApp textarea::placeholder {
            color: var(--app-muted);
            opacity: 1;
        }
        .hero { padding: 1.7rem 1.9rem; border-radius: 18px; background: linear-gradient(120deg,rgba(15,35,58,.97),rgba(18,69,133,.94)); color:#f8fafc; border:1px solid rgba(125,211,252,.28); box-shadow: 0 18px 55px rgba(2,8,23,.34); }
        .hero h1 { margin:0; font-size:2.5rem; letter-spacing:-.04em; color:#f8fafc; }
        .hero p { margin:.55rem 0 0; color:#dbeafe; }
        .glass-card { border-radius:16px; padding:1.2rem 1.3rem; background:rgba(15,23,42,.8); border:1px solid rgba(148,163,184,.18); }
        .login-panel { padding: 1.6rem 1.8rem; border-radius: 22px; background: linear-gradient(145deg,rgba(18,34,56,.98),rgba(13,25,43,.98)); border:1px solid rgba(125,211,252,.22); box-shadow: 0 25px 60px rgba(2,8,23,.42); color:#f1f5f9; }
        .login-panel p { color:#cbd5e1 !important; }
        .result-card { border-radius:16px; padding:1.25rem 1.5rem; border:1px solid rgba(148,163,184,.24); background:#111e31; color:#f1f5f9; }
        .result-label { font-size:1.65rem; font-weight:800; margin:.15rem 0; }
        .result-confidence { font-size:2.6rem; font-weight:850; line-height:1.1; }
        .ok { color:#34d399; } .defective { color:#fb7185; }
        .section-title { margin-top:1.5rem; padding-bottom:.4rem; border-bottom:1px solid var(--app-border); color:var(--app-text); }
        [data-testid="stFileUploader"] { border-radius:14px; }
        .login-title { font-size: 2.2rem; font-weight: 800; letter-spacing: -.04em; }
        .feature-list li { margin-bottom: .6rem; }
        .stApp [data-testid="stFileUploader"] section {
            background:var(--app-surface);
            border-color:var(--app-border);
        }
        .stApp [data-testid="stForm"] {
            background:var(--app-surface);
            border:1px solid var(--app-border);
            border-radius:16px;
            padding:1rem;
        }
        </style>
        """
        .replace("__BACKGROUND_IMAGE__", background_image)
        .replace("__TEXT__", palette["text"])
        .replace("__MUTED__", palette["muted"])
        .replace("__SURFACE__", palette["surface"])
        .replace("__INPUT__", palette["input"])
        .replace("__BORDER__", palette["border"])
        .replace("__OVERLAY__", palette["overlay"]),
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def runtime() -> tuple[object, int]:
    config = load_config(ROOT / "config.yaml")
    model_path = ROOT / config["training"]["model_path"]
    image_size = int(config["data"]["image_size"])
    return load_model(model_path), image_size


def png_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def render_login_page() -> None:
    app_theme()
    st.markdown(
        '<div class="hero"><h1>🏭 Casting Defect Inspector</h1>'
        '<p>Production-grade defect detection and explainable AI for smart manufacturing.</p></div>',
        unsafe_allow_html=True,
    )
    st.toggle("Light theme", key="light_theme", help="Switch between light and dark appearance.")

    left_col, right_col = st.columns([1.1, 1], gap="large")

    with left_col:
        with st.container(border=True):
            st.subheader("Project overview")
            st.markdown(
                """
                - Live casting inspection for pump impeller quality control
                - Binary defect classification using transfer learning
                - Grad-CAM visual explanations for operator review
                - Inspection tracking with session history and export
                """
            )
            st.markdown("### Production workflow")
            st.markdown(
                """
                1. Upload a casting image from a production line
                2. Run the trained AI model and confidence scoring
                3. Review the Grad-CAM hotspot overlay
                4. Save the result to the inspection log for auditability
                """
            )
            st.markdown("### Quality metrics")
            metrics = st.container(horizontal=True)
            with metrics:
                st.metric("Model", "EfficientNet", border=True)
                st.metric("Target", ">99%", border=True)
                st.metric("Dataset", "casting_512x512", border=True)

    with right_col:
        with st.container():
            st.markdown(
                '<div class="login-panel"><div class="login-title">Operator sign in</div>\n'
                '<p style="color:#cbd5e1; margin-top:0.5rem;">Access the inspection dashboard.</p></div>',
                unsafe_allow_html=True,
            )
            form = st.form("login_form")
            username = form.text_input("Username", key="login_username")
            password = form.text_input("Password", type="password", key="login_password")
            submit = form.form_submit_button("Sign in", type="primary")

            if submit:
                users = get_demo_users()
                if username in users and users[username] == password:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

            st.caption(
                "Demo login: username admin123, password admin123. "
                "You can also use admin / admin123 or inspector / factory2025."
            )


def render_dashboard() -> None:
    history_data = load_history()
    history_df = pd.DataFrame(history_data)
    total_inspections = len(history_data)
    defect_count = sum(item["prediction"] == "Defective" for item in history_data)
    defect_rate = defect_count / total_inspections if total_inspections else None
    pass_rate = 1 - defect_rate if defect_rate is not None else None
    average_confidence = (
        sum(float(item["confidence"]) for item in history_data) / total_inspections
        if total_inspections
        else None
    )

    st.markdown(
        '<div class="hero"><h1>🏭 Casting Defect Inspector</h1>'
        '<p>Quality control dashboard for submersible pump impeller inspection.</p></div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.title("Factory control")
        st.toggle("Light theme", key="light_theme", help="Switch between light and dark appearance.")
        app_theme()
        st.caption(f"Signed in as: {st.session_state.username}")
        st.button("Log out", on_click=lambda: setattr(st.session_state, "authenticated", False))
        st.markdown("---")
        st.subheader("Project status")
        st.metric("Shift", "A", border=True)
        st.metric("Line", "Impeller-14", border=True)
        st.metric("Model state", "Ready", border=True)

    overview, inspect, history, reports = st.tabs(["Overview", "Inspection", "History", "Reports"])

    with overview:
        kpi_grid = st.container(horizontal=True)
        with kpi_grid:
            st.metric("Images reviewed", total_inspections, border=True)
            st.metric(
                "Defect rate",
                f"{defect_rate:.1%}" if defect_rate is not None else "—",
                border=True,
            )
            st.metric(
                "Average confidence",
                f"{average_confidence:.1%}" if average_confidence is not None else "—",
                border=True,
            )
            st.metric(
                "Pass rate",
                f"{pass_rate:.1%}" if pass_rate is not None else "—",
                border=True,
            )

        col_a, col_b = st.columns(2)
        with col_a:
            with st.container(border=True):
                st.subheader("Production insights")
                st.markdown(
                    """
                    - Human review is recommended when confidence sits between 70% and 85%
                    - Defect hotspots are most frequent near the rim and hub regions
                    - Line calibration is stable; anomaly rate remains within control limits
                    """
                )
        with col_b:
            with st.container(border=True):
                st.subheader("Inspection workflow")
                st.progress(0.7, text="AI pipeline ready")
                st.write("Dataset: casting_512x512")
                st.write("Model: EfficientNet transfer learning")
                st.write("Grad-CAM: active for explainability")

        with st.container(border=True):
            st.subheader("Next actions")
            st.markdown(
                """
                - Add operator feedback loop for false positive review
                - Expand training to include more defect families
                - Schedule weekly model validation on fresh samples
                """
            )

    with inspect:
        render_inspection_tab()

    with history:
        if not history_data:
            st.info(
                "No inspections have been saved yet. Open the Inspection tab, upload an "
                "image, and select “Save inspection to history” after the model result appears."
            )
        else:
            st.dataframe(history_df, width="stretch", hide_index=True)
            csv_data = history_df.to_csv(index=False).encode("utf-8")
            json_data = history_df.to_json(orient="records", indent=2)
            download_left, download_right = st.columns(2)
            with download_left:
                st.download_button(
                    "Download history CSV",
                    data=csv_data,
                    file_name="inspection_history.csv",
                    mime="text/csv",
                    width="stretch",
                )
            with download_right:
                st.download_button(
                    "Download history JSON",
                    data=json_data,
                    file_name="inspection_history.json",
                    mime="application/json",
                    width="stretch",
                )

    with reports:
        if not history_data:
            st.info(
                "No report data is available yet. Save an inspection first; summary metrics, "
                "prediction counts, and report downloads will appear here."
            )
        else:
            st.subheader("Inspection summary")
            report_metrics = st.container(horizontal=True)
            with report_metrics:
                st.metric("Total inspections", total_inspections, border=True)
                st.metric("Defective", defect_count, border=True)
                st.metric(
                    "Average confidence",
                    f"{average_confidence:.1%}",
                    border=True,
                )

            col_counts, col_confidence = st.columns(2)
            with col_counts:
                st.subheader("Prediction counts")
                prediction_counts = history_df["prediction"].value_counts().rename_axis("Prediction")
                st.bar_chart(prediction_counts, x_label="Prediction", y_label="Inspections")
            with col_confidence:
                st.subheader("Confidence by inspection")
                confidence_data = history_df[["timestamp", "confidence"]].copy()
                confidence_data["confidence"] *= 100
                st.line_chart(
                    confidence_data,
                    x="timestamp",
                    y="confidence",
                    x_label="Inspection time",
                    y_label="Confidence (%)",
                )

            st.subheader("Recent inspection records")
            st.dataframe(
                history_df[["timestamp", "file_name", "prediction", "confidence", "username"]],
                width="stretch",
                hide_index=True,
            )
            st.download_button(
                "Download project report (CSV)",
                data=history_df.to_csv(index=False).encode("utf-8"),
                file_name="casting_inspection_report.csv",
                mime="text/csv",
                width="stretch",
            )


def render_inspection_tab() -> None:
    image_source = st.segmented_control(
        "Image source",
        options=["Upload image", "Webcam / production camera"],
        default="Upload image",
        key="inspection_image_source",
    )

    if image_source == "Webcam / production camera":
        st.caption(
            "Use a webcam or USB production camera available to this browser. "
            "Camera access requires permission; deployed apps must use HTTPS."
        )
        captured = st.camera_input(
            "Capture a casting image",
            key="inspection_camera_capture",
            help="Frame one casting part and take a snapshot to run AI inspection.",
            resolution="720p",
        )
        image_file = captured
        file_name = getattr(captured, "name", None) or "camera_capture.jpg"
        source_description = "camera capture"
    else:
        image_file = st.file_uploader(
            "Upload a grayscale casting image",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Supported formats: JPG, PNG, BMP. The model normalizes the image before inspection.",
            key="inspection_image_upload",
        )
        file_name = getattr(image_file, "name", None) or "uploaded_image"
        source_description = "uploaded image"

    if not image_file:
        st.info(
            "Capture or upload an image to start an inspection. "
            "Saved results will appear in History and Reports."
        )
        return

    try:
        image = Image.open(image_file).convert("L")
        image.load()
    except (OSError, ValueError) as exc:
        st.error(f"Unable to read this {source_description}. Please capture a valid image. ({exc})")
        return

    with st.spinner("Running inspection and generating Grad-CAM explanation..."):
        try:
            model, image_size = runtime()
            batch = preprocess_pil_image(image, image_size)
            result = predict(model, batch)
            heatmap = make_gradcam(model, batch, class_index=result["class_index"])
            heatmap_rgb, overlay = make_visuals(np.asarray(image), heatmap)
        except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
            st.error(f"Inspection could not be completed: {exc}")
            return

    label = result["label"]
    confidence = float(result["confidence"])
    accent = "defective" if label == "Defective" else "ok"
    icon = "⚠️" if label == "Defective" else "✅"

    left, right = st.columns([1, 2])
    with left:
        st.markdown(
            f'<div class="result-card"><div>Classification</div>'
            f'<div class="result-label {accent}">{icon} {label}</div>'
            f'<div class="result-confidence {accent}">{confidence:.1%}</div>'
            '<div>model confidence</div></div>',
            unsafe_allow_html=True,
        )
    with right:
        st.subheader("Model confidence")
        st.progress(confidence, text=f"{confidence:.1%} confidence")
        st.bar_chart(
            {"Probability": {"Defective": result["defect_probability"], "OK": result["ok_probability"]}},
            horizontal=True,
            height=150,
        )

    st.markdown('<h2 class="section-title">Explainable AI (Grad-CAM)</h2>', unsafe_allow_html=True)
    columns = st.columns(3, gap="medium")
    columns[0].image(image, caption="Original image", width="stretch")
    columns[1].image(heatmap_rgb, caption="Grad-CAM heatmap (jet)", width="stretch")
    columns[2].image(overlay, caption="Grad-CAM overlay", width="stretch")
    st.caption("Warmer regions indicate image areas that contributed more strongly to the selected prediction.")

    note = st.text_area("Operator comments", placeholder="Add notes for the quality review team...")
    if st.button("Save inspection to history"):
        entry = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "file_name": file_name,
            "prediction": label,
            "confidence": round(confidence, 6),
            "defect_probability": round(float(result["defect_probability"]), 6),
            "ok_probability": round(float(result["ok_probability"]), 6),
            "operator_note": note,
            "username": st.session_state.username,
        }
        try:
            save_inspection(entry)
        except sqlite3.Error as exc:
            st.error(f"Could not save this inspection to history: {exc}")
        else:
            st.success("Inspection saved to persistent project history.")
            st.rerun()

    report = {
        "file_name": file_name,
        "prediction": label,
        "confidence": round(confidence, 6),
        "defective_probability": round(float(result["defect_probability"]), 6),
        "ok_probability": round(float(result["ok_probability"]), 6),
    }
    download_left, download_right = st.columns(2)
    with download_left:
        st.download_button(
            "Download Grad-CAM overlay",
            data=png_bytes(overlay),
            file_name=f"{Path(file_name).stem}_gradcam_overlay.png",
            mime="image/png",
            width="stretch",
        )
    with download_right:
        st.download_button(
            "Download inspection report",
            data=json.dumps(report, indent=2),
            file_name=f"{Path(file_name).stem}_inspection_report.json",
            mime="application/json",
            width="stretch",
        )


initialize_session()
initialize_history()

if not st.session_state.authenticated:
    render_login_page()
    st.stop()

render_dashboard()
