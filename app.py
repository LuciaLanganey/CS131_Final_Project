"""Streamlit UI for the clothing describer."""

import html
import os
import tempfile

import streamlit as st

from phase4.describe import describe_image, load_models


def format_label(label: str) -> str:
    return label.replace("_", " ").title()


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
    "Upload a photo of a garment to get a natural-language description "
    "of its pattern, sleeve style, silhouette, season, and type."
)

uploaded = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    help="On mobile, you can take a new photo or pick one from your library.",
)

if uploaded is not None:
    st.caption("Visual preview of your uploaded garment.")
    st.image(uploaded, caption="Uploaded garment photo", use_container_width=True)

    suffix = os.path.splitext(uploaded.name)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded.getvalue())
        image_path = tmp.name

    with st.status("Analyzing garment...", expanded=True) as status:
        classifiers, encoders = load_models()
        description, predictions = describe_image(image_path, classifiers, encoders)
        status.update(label="Analysis complete", state="complete", expanded=False)

    render_results(description, predictions)
