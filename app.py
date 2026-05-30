"""Streamlit UI for the clothing describer."""

import streamlit as st

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
