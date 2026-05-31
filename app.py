"""Streamlit UI for the clothing describer."""

import html
import os
import tempfile

import streamlit as st

from phase4.describe import describe_image, load_models


@st.cache_resource
def get_models():
    return load_models()


def format_label(label: str) -> str:
    return label.replace("_", " ").title()


def render_error(message: str) -> None:
    st.error(message)


def render_results(description: str, predictions: dict) -> None:
    attribute_rows = "".join(
        f"<dt>{html.escape(format_label(label))}</dt>"
        f"<dd>{html.escape(str(value))}</dd>"
        for label, value in predictions.items()
    )

    st.markdown(
        f"""
<div aria-live="polite" aria-atomic="true" aria-label="Analysis results">
  <p><strong>Analysis complete.</strong></p>
  <h2>Description</h2>
  <p>{html.escape(description)}</p>
  <h2>Results</h2>
  <dl>
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

st.title("Clothing Describer")
st.write(
    "Take or upload a photo of a garment to get a natural-language description "
    "of its pattern, sleeve style, silhouette, season, and type."
)

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
    caption = "Photo from camera" if from_camera else "Uploaded garment photo"
    st.caption("Visual preview of your garment.")
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
