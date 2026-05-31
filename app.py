"""Streamlit UI for the clothing describer."""

import html
import os
import tempfile

import streamlit as st

from phase4.describe import describe_image, load_models


@st.cache_resource
def get_models():
    return load_models()


def inject_styles() -> None:
    st.markdown(
        """
<style>
    :root,
    .stApp[data-theme="light"] {
        --bg-soft: #f7f5f2;
        --card: #ffffff;
        --text: #1f1f1f;
        --muted: #6b6560;
        --accent: #3d5a4c;
        --accent-light: #e8efeb;
        --border: #e3ded8;
        --shadow: 0 8px 24px rgba(31, 31, 31, 0.06);
    }

    .stApp[data-theme="dark"] {
        --bg-soft: #2a302c;
        --card: #1c211e;
        --text: #f2f2f2;
        --muted: #b5aea8;
        --accent: #8fc4a8;
        --accent-light: #243329;
        --border: #3a433e;
        --shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    }

    @media (prefers-color-scheme: dark) {
        .stApp:not([data-theme="light"]) {
            --bg-soft: #2a302c;
            --card: #1c211e;
            --text: #f2f2f2;
            --muted: #b5aea8;
            --accent: #8fc4a8;
            --accent-light: #243329;
            --border: #3a433e;
            --shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        }
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 760px;
    }

    .app-hero {
        background: linear-gradient(135deg, var(--accent-light) 0%, var(--bg-soft) 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.75rem 1.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow);
        text-align: center;
    }

    .stMarkdown div.app-hero-title,
    div[data-testid="stMarkdownContainer"] div.app-hero-title {
        color: var(--text) !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
        line-height: 1.2 !important;
        margin: 0 0 0.75rem 0 !important;
    }

    .stMarkdown div.app-hero-subtitle,
    div[data-testid="stMarkdownContainer"] div.app-hero-subtitle {
        color: var(--muted) !important;
        font-size: 1.15rem !important;
        line-height: 1.6 !important;
        margin: 0 !important;
    }

    .stMarkdown p.section-label,
    div[data-testid="stMarkdownContainer"] p.section-label {
        color: var(--accent) !important;
        font-size: 0.85rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.08em;
        margin: 0 0 0.75rem 0 !important;
        text-transform: uppercase;
    }

    .app-hero,
    .results-panel,
    .description-text,
    .attributes-grid > div,
    div[data-testid="stImage"],
    div[data-testid="stFileUploader"],
    div[data-testid="stCameraInput"] {
        color: var(--text);
    }

    div[data-testid="stImage"] {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.75rem;
        box-shadow: var(--shadow);
    }

    div[data-testid="stImage"] img {
        border-radius: 10px;
    }

    div[data-testid="stFileUploader"] {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem 1rem 1rem;
        text-align: center;
    }

    div[data-testid="stCameraInput"] {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem;
        text-align: center;
        overflow: visible;
    }

    div[data-testid="stCameraInput"] > div {
        overflow: visible;
    }

    div[data-testid="stCameraInput"] [data-testid="baseButton-secondary"] {
        align-items: center !important;
        background-color: var(--bg-soft) !important;
        border: 1px dashed var(--border) !important;
        color: var(--text) !important;
        display: flex !important;
        justify-content: center !important;
        margin-left: auto;
        margin-right: auto;
        min-height: 140px;
        width: 100%;
    }

    div[data-testid="stCameraInput"] [data-testid="baseButton-secondary"] p,
    div[data-testid="stCameraInput"] [data-testid="baseButton-secondary"] span,
    div[data-testid="stCameraInput"] [data-testid="baseButton-secondary"] div {
        color: var(--text) !important;
    }

    div[data-testid="stCameraInput"] label,
    div[data-testid="stCameraInput"] p,
    div[data-testid="stCameraInput"] small {
        color: var(--text);
        justify-content: center;
        text-align: center;
        width: 100%;
    }

    div[data-testid="stCameraInput"] video,
    div[data-testid="stCameraInput"] img {
        max-height: 180px;
        width: 100%;
        object-fit: cover;
        border-radius: 10px;
        display: block;
        margin: 0 auto;
    }

    div[data-testid="stStatusWidget"] {
        border-radius: 12px;
    }

    .results-panel {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 16px;
        box-sizing: border-box;
        margin-top: 1rem;
        padding: 1.5rem;
        box-shadow: var(--shadow);
        width: 100%;
    }

    div[data-testid="stMarkdownContainer"] .results-panel,
    div[data-testid="stMarkdownContainer"] .results-top,
    div[data-testid="stMarkdownContainer"] .description-text,
    div[data-testid="stMarkdownContainer"] .results-heading {
        text-align: center !important;
    }

    .results-panel h2 a,
    .results-top h2 a {
        display: none !important;
    }

    .results-top {
        align-items: stretch;
        display: flex;
        flex-direction: column;
        margin-bottom: 1.5rem;
        text-align: center;
        width: 100%;
    }

    .stMarkdown div.results-heading,
    div[data-testid="stMarkdownContainer"] div.results-heading {
        color: var(--text) !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        margin: 0 0 0.75rem 0 !important;
        padding: 0;
        width: 100%;
    }

    .stMarkdown div.description-text,
    div[data-testid="stMarkdownContainer"] div.description-text {
        align-self: stretch;
        background: var(--accent-light);
        border: 1px solid var(--accent);
        border-radius: 12px;
        box-sizing: border-box;
        color: var(--text) !important;
        display: block;
        font-size: 1.35rem !important;
        line-height: 1.65 !important;
        margin: 0 !important;
        padding: 1.1rem 1.25rem;
        text-align: center;
        width: 100%;
    }

    .stMarkdown p.status-badge,
    div[data-testid="stMarkdownContainer"] p.status-badge {
        align-self: center;
        background: var(--accent-light);
        border-radius: 999px;
        color: var(--accent) !important;
        display: block;
        font-size: 0.85rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.04em;
        margin: 0 0 1rem 0 !important;
        padding: 0.35rem 0.75rem;
        text-transform: uppercase;
        width: fit-content;
    }

    .results-panel p {
        margin-left: 0;
        margin-right: 0;
    }

    .results-panel > .results-heading {
        margin-bottom: 0.75rem;
    }

    .attributes-grid {
        display: grid;
        gap: 0.75rem;
        grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
        list-style: none;
        margin: 0;
        padding: 0;
    }

    .attributes-grid > div {
        background: var(--bg-soft);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.85rem 1rem;
        text-align: center;
    }

    .attributes-grid dt {
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        margin: 0 0 0.35rem 0;
        text-transform: uppercase;
    }

    .stMarkdown dl.attributes-grid dd,
    div[data-testid="stMarkdownContainer"] dl.attributes-grid dd {
        color: var(--text) !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        margin: 0;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }
</style>
""",
        unsafe_allow_html=True,
    )


def format_label(label: str) -> str:
    return label.replace("_", " ").title()


def render_error(message: str) -> None:
    st.error(message)


def render_results(description: str, predictions: dict) -> None:
    attribute_rows = "".join(
        f"<div><dt>{html.escape(format_label(label))}</dt>"
        f"<dd>{html.escape(str(value))}</dd></div>"
        for label, value in predictions.items()
    )

    st.markdown(
        f"""
<div class="results-panel" aria-live="polite" aria-atomic="true" aria-label="Analysis results">
  <div class="results-top">
    <p class="status-badge">Analysis complete</p>
    <div class="results-heading" role="heading" aria-level="2">Description</div>
    <div class="description-text">{html.escape(description)}</div>
  </div>
  <div class="results-heading" role="heading" aria-level="2">Attributes</div>
  <dl class="attributes-grid">
    {attribute_rows}
  </dl>
</div>
""",
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="Clothing Describer",
    page_icon="👕",
    layout="centered",
)

inject_styles()

st.markdown(
    """
<div class="app-hero">
  <div class="app-hero-title" role="heading" aria-level="1">Clothing Describer</div>
  <div class="app-hero-subtitle">
    Take or upload a photo of a garment to get a natural-language description
    of its pattern, sleeve style, silhouette, season, and type.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<p class="section-label">Add a photo</p>', unsafe_allow_html=True)

camera_photo = st.camera_input(
    "Take a photo",
    help="Opens your device camera. Works on most mobile browsers.",
)
uploaded = st.file_uploader(
    "Or choose an image",
    type=["jpg", "jpeg", "png"],
    help="On mobile, you can take a new photo or pick one from your library.",
)

from_camera = camera_photo is not None
image_source = camera_photo if from_camera else uploaded

if image_source is not None:
    st.markdown('<p class="section-label">Preview</p>', unsafe_allow_html=True)
    caption = "Photo from camera" if from_camera else "Uploaded garment photo"
    st.image(image_source, caption=caption, use_container_width=True)

    filename = getattr(image_source, "name", None) or "photo.jpg"
    suffix = os.path.splitext(filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(image_source.getvalue())
        image_path = tmp.name

    analysis_failed = False
    try:
        with st.status("Analyzing garment...", expanded=True) as status:
            try:
                classifiers, encoders = get_models()
                description, predictions = describe_image(
                    image_path, classifiers, encoders
                )
                status.update(
                    label="Analysis complete",
                    state="complete",
                    expanded=False,
                )
            except FileNotFoundError:
                status.update(
                    label="Analysis failed",
                    state="error",
                    expanded=True,
                )
                render_error(
                    "Trained models were not found. Run "
                    "`python3 phase4/classifier.py` first, then reload this page."
                )
                analysis_failed = True
            except ValueError:
                status.update(
                    label="Analysis failed",
                    state="error",
                    expanded=True,
                )
                render_error(
                    "Could not analyze this image. Try a clearer photo with "
                    "the garment centered and good lighting."
                )
                analysis_failed = True
            except Exception:
                status.update(
                    label="Analysis failed",
                    state="error",
                    expanded=True,
                )
                render_error(
                    "Something went wrong while analyzing the image. "
                    "Please try again with a different photo."
                )
                analysis_failed = True
    finally:
        if os.path.exists(image_path):
            os.unlink(image_path)

    if not analysis_failed:
        render_results(description, predictions)

inject_styles()
