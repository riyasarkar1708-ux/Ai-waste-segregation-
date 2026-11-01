import os
import gdown
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="AI Waste Segregation",
    page_icon="♻️",
    layout="wide"
)

MODEL_PATH = "Waste_Segregation_Model.keras"
DRIVE_ID = "1Yefyp0re81U92h5JaJMOTDK8QdNoQiy1"

CLASS_NAMES = ["E-Waste", "Metal", "Organic", "Paper", "Plastic"]

TIPS = {
    "E-Waste": "Take electronic waste to an authorised e-waste collection centre.",
    "Metal": "Separate metal items for recycling.",
    "Organic": "Place suitable food waste in a composting bin.",
    "Paper": "Keep paper clean and dry for recycling.",
    "Plastic": "Separate recyclable plastic according to local guidelines."
}

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={DRIVE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

    if not os.path.exists(MODEL_PATH):
        st.error("Could not download the model. Check Google Drive sharing settings.")
        st.stop()

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

st.title("♻️ AI Waste Segregation System")
st.write("Upload a waste image to identify its category.")

uploaded = st.file_uploader(
    "Choose a waste image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", width=350)

    if st.button("Classify Waste"):
        with st.spinner("Analysing image..."):
            model = load_model()
            resized = image.resize((160, 160))
            array = np.array(resized).astype("float32")
            array = np.expand_dims(array, axis=0)

            predictions = model.predict(array, verbose=0)[0]
            index = int(np.argmax(predictions))
            category = CLASS_NAMES[index]

        st.success(f"Predicted category: {category}")
        st.metric("Model confidence", f"{predictions[index]*100:.2f}%")
        st.info("Disposal suggestion: " + TIPS[category])

st.divider()
st.subheader("Five Waste Categories")
st.write("🍃 Organic | ♻️ Plastic | 📄 Paper | 🔩 Metal | 💻 E-Waste")
st.caption("Class XII Artificial Intelligence Project")
