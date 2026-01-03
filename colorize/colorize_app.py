import streamlit as st
import numpy as np
import cv2
from pathlib import Path

# ================= PAGE CONFIG (ONLY ONCE) =================
st.set_page_config(
    page_title="Medical Image Colorization",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================= SAFE SUPER-RESOLUTION LOADER =================
def load_superres(model_path):
    """
    Safely load DNN Super Resolution.
    Works even if opencv-contrib is missing or Python 3.13 breaks it.
    """
    try:
        if hasattr(cv2, "dnn_superres"):
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel(str(model_path))
            sr.setModel("espcn", 2)
            return sr
    except Exception as e:
        st.warning("⚠️ Super-resolution disabled (OpenCV limitation).")
    return None

# ================= UI STYLING =================
st.markdown("""
<style>
body {
    background: radial-gradient(circle at center, #e0f7fa, #80deea);
}
.main {
    background-color: rgba(255, 255, 255, 0.85);
    border-radius: 10px;
    padding: 30px;
}
h1, h2, h3 { color: black; }
</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.markdown("## 🧬 Medical Image Colorizer (Research Platform)")

st.markdown("""
**Medical Image Colorization** converts grayscale medical scans into
color-enhanced images using **deep learning (Caffe model)**.

> ⚠️ For **medical & research use only**
""")

# ================= PATHS =================
DIR = Path(__file__).parent
PROTOTXT = DIR / "model" / "colorization_deploy_v2.prototxt"
POINTS = DIR / "model" / "pts_in_hull.npy"
MODEL = DIR / "model" / "colorization_release_v2.caffemodel"
SR_MODEL_PATH = DIR / "model" / "ESPCN_x2.pb"

# ================= LOAD COLORIZATION MODEL =================
net = cv2.dnn.readNetFromCaffe(str(PROTOTXT), str(MODEL))

pts = np.load(POINTS)
pts = pts.transpose().reshape(2, 313, 1, 1)

class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")

net.getLayer(class8).blobs = [pts.astype(np.float32)]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype=np.float32)]

# ================= COLORIZE FUNCTION =================
def colorize_patch(L):
    L = L.astype("float32") - 50
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0].transpose((1, 2, 0))
    return ab

# ================= LOAD SUPER RESOLUTION (SAFE) =================
sr = load_superres(SR_MODEL_PATH)

# ================= PATCH PARAMETERS =================
TILE_SIZE = 224
OVERLAP = 112

# ================= PATCH PROCESSING =================
def process_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    h, w = L.shape

    ab_accum = np.zeros((h, w, 2), np.float32)
    weight = np.zeros((h, w, 2), np.float32)

    for y in range(0, h, TILE_SIZE - OVERLAP):
        for x in range(0, w, TILE_SIZE - OVERLAP):
            y2, x2 = min(y + TILE_SIZE, h), min(x + TILE_SIZE, w)
            L_crop = cv2.resize(L[y:y2, x:x2], (TILE_SIZE, TILE_SIZE))
            ab = colorize_patch(L_crop)
            ab = cv2.resize(ab, (x2 - x, y2 - y))

            mask = np.ones_like(ab)
            ab_accum[y:y2, x:x2] += ab * mask
            weight[y:y2, x:x2] += mask

    ab_final = ab_accum / np.maximum(weight, 1e-8)
    colorized = np.concatenate((L[:, :, None], ab_final), axis=2)
    bgr = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    return np.clip(bgr, 0, 255).astype(np.uint8)

# ================= FILE UPLOAD =================
uploaded_files = st.file_uploader(
    "Upload medical grayscale images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        data = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)

        st.write(f"Processing **{file.name}**...")
        colorized = process_image(img)

        if sr is not None:
            colorized = sr.upsample(colorized)

        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
        with col2:
            st.image(cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB), caption="Colorized", use_container_width=True)

        _, buffer = cv2.imencode(".jpg", colorized)
        st.download_button(
            "Download Result",
            buffer.tobytes(),
            file_name=f"colorized_{file.name}",
            mime="image/jpeg"
        )
