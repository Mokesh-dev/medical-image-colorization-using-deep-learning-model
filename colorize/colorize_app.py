import streamlit as st
import numpy as np
import cv2
from pathlib import Path
from cv2 import dnn_superres

# === Page Config ===
st.set_page_config(
    page_title="Medical Image Colorization",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
    <style>
    body {
        background: radial-gradient(circle at center, #e0f7fa, #80deea);
        animation: pulseBG 8s ease-in-out infinite alternate;
    }

    @keyframes pulseBG {
        0% {
            background: radial-gradient(circle at center, #e0f7fa, #80deea);
        }
        100% {
            background: radial-gradient(circle at center, #b2ebf2, #4dd0e1);
        }
    }

    .main {
        background-color: rgba(255, 255, 255, 0.85);
        border-radius: 10px;
        padding: 30px;
        box-shadow: 0 0 20px rgba(0,0,0,0.1);
    }

    h1, h2, h3 {
        color: #000000; /* Title black as you asked */
        font-family: Arial, sans-serif;
    }

    .emoji-spin {
        display: inline-block;
        animation: spin 3s linear infinite;
    }

    .emoji-wave {
        display: inline-block;
        animation: wave 2s infinite;
        transform-origin: 70% 70%;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    @keyframes wave {
        0% { transform: rotate( 0.0deg) }
       10% { transform: rotate(14.0deg) }  
       20% { transform: rotate(-8.0deg) }
       30% { transform: rotate(14.0deg) }
       40% { transform: rotate(-4.0deg) }
       50% { transform: rotate(10.0deg) }
       60% { transform: rotate( 0.0deg) }  
      100% { transform: rotate( 0.0deg) }
    }
    </style>
""", unsafe_allow_html=True)

# === Welcome ===
st.markdown("""
    <div style="text-align: left; margin-bottom: 5px;">
        <h2 style="font-size: 2.5em; font-family: Arial, sans-serif; margin: 0;">
            <span class="emoji-wave"></span>Welcome To,
        </h2>
    </div>
""", unsafe_allow_html=True)

# === Title ===
st.markdown("""
    <h2 style="font-size: 2.5em; font-family: Arial, sans-serif; margin: 0;">
        <span class="emoji-spin">🧬</span> Medical Image Colorizer build-in-platform 
    </h2>
""", unsafe_allow_html=True)
# === Introduction Headline ===
st.markdown("## 📌 Introduction")

# === Introduction Description ===
st.markdown("""
**Medical Image Colorization** is a specialized tool built using **Deep Learning Models**
to convert black-and-white medical scans into realistic colorized images which is fully automated in click and view process.
This improves visual clarity for medical staff and **builds trust and confidence**
between patients, doctors, hospitals, and management.

This tool is designed exclusively for **medical professionals and researchers**
to support better diagnostics, transparent patient communication,
and secure record keeping.
""")

# === Disclaimer ===
st.markdown("""
> ❗ **Disclaimer:** This tool is machine based models which is strictly for **medical and research purposes** only.It is **not intended for general public use** or non-professional applications.
""")

st.markdown("""
 **Hospitals** 
             For keeping better medical records & reports.  
 **Doctors / Radiologists** 
             To enhance visibility of scans for diagnostics & teaching.  

""")


st.markdown("""
> ⚠️ **Note:** The model generates a best-possible colorized version and may not perfectly replicate real colors.""")


# === Load Models ===
st.write("Models are loaded. Waiting for input...")


DIR = Path(__file__).parent

PROTOTXT = DIR / "model" / "colorization_deploy_v2.prototxt"
POINTS = DIR / "model" / "pts_in_hull.npy"
MODEL = DIR / "model" / "colorization_release_v2.caffemodel"
SR_MODEL_PATH = DIR / "model" / "ESPCN_x2.pb"

net = cv2.dnn.readNetFromCaffe(str(PROTOTXT), str(MODEL))
pts = np.load(POINTS)
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype(np.float32)]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype=np.float32)]

# === Colorize a single patch ===
def colorize_patch(L_patch):
    L_patch = L_patch.astype("float32") - 50
    net.setInput(cv2.dnn.blobFromImage(L_patch))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    return ab

# === Super-Resolution ===
sr = dnn_superres.DnnSuperResImpl_create()
sr.readModel(str(SR_MODEL_PATH))
sr.setModel("espcn", 2)

# === Session state for processed images ===
if "processed_images" not in st.session_state:
    st.session_state.processed_images = {}

# === Patch Parameters ===
TILE_SIZE = 224
OVERLAP = 112

# === Page Config ===
st.set_page_config(
    page_title="Medical Image Colorization",
    layout="wide",   # ✅ enable wide mode
    initial_sidebar_state="expanded",
)

# === Sidebar Controls ===
blend_mode = st.sidebar.radio("⚙️ Blending Modes setting""", ["Linear Feather", "Gaussian Feather"])
feather_size = st.sidebar.slider("Feather Size (px)", 5, 100, 30)  # min=5, max=100, default=30

# === Mask Creation Functions ===
def create_linear_mask(h, w, feather=32):
    """Linear feather mask with smooth edges"""
    mask = np.ones((h, w), np.float32)
    y = np.linspace(0, 1, h)
    x = np.linspace(0, 1, w)
    xv, yv = np.meshgrid(x, y)

    # Distance from borders
    mask_x = np.minimum(xv, 1 - xv)
    mask_y = np.minimum(yv, 1 - yv)
    mask = np.minimum(mask_x, mask_y)

    # Apply feathering
    mask = np.clip(mask * (w / (2 * feather)), 0, 1)
    return cv2.merge([mask, mask])

def create_gaussian_mask(h, w, feather=32):
    """Gaussian decay mask for ultra-smooth blending"""
    xv, yv = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    d = np.sqrt(xv**2 + yv**2)
    sigma = feather / max(h, w)
    mask = np.exp(-(d**2) / (2 * sigma**2))
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return cv2.merge([mask, mask])

# === Patch-based Colorization with Blending ===
def process_image_in_patches(image):
    scaled = image.astype(np.float32) / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    L_orig = cv2.split(lab)[0]
    h, w = L_orig.shape

    colorized_ab = np.zeros((h, w, 2), dtype=np.float32)
    weight_mask = np.zeros((h, w, 2), dtype=np.float32)

    # === Progress Bar ===
    progress_bar = st.progress(0)
    percent_text = st.empty()
    total_steps = len(range(0, h, TILE_SIZE - OVERLAP)) * len(range(0, w, TILE_SIZE - OVERLAP))
    step = 0

    for y in range(0, h, TILE_SIZE - OVERLAP):
        for x in range(0, w, TILE_SIZE - OVERLAP):
            y1, y2 = y, min(y + TILE_SIZE, h)
            x1, x2 = x, min(x + TILE_SIZE, w)

            L_crop = L_orig[y1:y2, x1:x2]
            L_resized = cv2.resize(L_crop, (TILE_SIZE, TILE_SIZE))

            ab = colorize_patch(L_resized)
            ab = cv2.resize(ab, (x2 - x1, y2 - y1))

            # === Choose blending mask ===
            if blend_mode == "Linear Feather":
                mask = create_linear_mask(ab.shape[0], ab.shape[1], feather=feather_size)
            else:
                mask = create_gaussian_mask(ab.shape[0], ab.shape[1], feather=feather_size)

            colorized_ab[y1:y2, x1:x2, :] += ab * mask
            weight_mask[y1:y2, x1:x2, :] += mask

            step += 1
            percent_complete = int((step / total_steps) * 100)
            progress_bar.progress(percent_complete)
            percent_text.markdown(f"**Processing: {percent_complete}%**")

    percent_text.markdown("✅ **Processing complete!**")

    ab_final = colorized_ab / np.maximum(weight_mask, 1e-8)
    colorized_lab = np.concatenate((L_orig[:, :, np.newaxis], ab_final), axis=2)
    colorized_bgr = cv2.cvtColor(colorized_lab, cv2.COLOR_LAB2BGR)
    colorized_bgr = np.clip(colorized_bgr, 0, 1)
    colorized_bgr = (255 * colorized_bgr).astype(np.uint8)

    return colorized_bgr

# === Upload & Process ===
uploaded_files = st.file_uploader(
    "Upload medical B/W images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.divider()
        file_name = uploaded_file.name

        # === Check cache ===
        if file_name in st.session_state.processed_images:
            original, final_upscaled = st.session_state.processed_images[file_name]
        else:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            st.write("Processing with patch-based colorization...")
            colorized_patched = process_image_in_patches(image)
            final_upscaled = sr.upsample(colorized_patched)

            # Cache result
            st.session_state.processed_images[file_name] = (image, final_upscaled)
            original = image

        st.subheader(f"Result: {file_name}")
        col1, col2 = st.columns(2)

        with col1:
            st.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            st.markdown("**Original**")

        with col2:
            st.image(cv2.cvtColor(final_upscaled, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            st.markdown("**Colorized + Upscaled**")

        is_success, buffer = cv2.imencode(".jpg", final_upscaled)
        if is_success:
            st.download_button(
                label="Download Colorized",
                data=buffer.tobytes(),
                file_name=f"patch_colorized_{file_name}",
                mime="image/jpeg"
            )


