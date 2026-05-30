"""Streamlit UI for the clothing describer."""

import os
import tempfile

import streamlit as st

from phase4.describe import describe_image, load_models

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
    st.image(uploaded, caption="Uploaded image", use_container_width=True)

    suffix = os.path.splitext(uploaded.name)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded.getvalue())
        image_path = tmp.name

    with st.spinner("Analyzing garment..."):
        classifiers, encoders = load_models()
        description, predictions = describe_image(image_path, classifiers, encoders)

    st.subheader("Description")
    st.write(description)

    for label, value in predictions.items():
        st.write(f"**{label.replace('_', ' ').title()}:** {value}")
