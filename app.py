import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2

from model.efficientnet_b4 import EfficientNetB4
from model.efficientnet_b4_gray import EfficientNetB4 as EfficientNetGray

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_rgb_model():
    model = EfficientNetB4(num_classes=2, pretrained=False)
    state = torch.load("RGB.pth", map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model

@st.cache_resource
def load_freq_model():
    model = EfficientNetGray(num_classes=2, pretrained=False, in_channels=1)
    state = torch.load("frequency.pth", map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model

rgb_model = load_rgb_model()
freq_model = load_freq_model()


# -----------------------------
# Face Detection (Error if NO face)
# -----------------------------
def detect_face(image_pil):
    img = np.array(image_pil)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    return len(faces) > 0


# -----------------------------
# Preprocessing
# -----------------------------
import torchvision.transforms as transforms

rgb_transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
])

freq_transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.Grayscale(),
    transforms.ToTensor()
])


def convert_to_frequency(image_pil):
    img = np.array(image_pil)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1)

    magnitude = magnitude / (magnitude.max() + 1e-8) * 255
    magnitude = magnitude.astype(np.uint8)

    return Image.fromarray(magnitude)


# -----------------------------
# Prediction
# -----------------------------
def predict_image(image_pil):

    # RGB branch
    rgb_tensor = rgb_transform(image_pil).unsqueeze(0)
    rgb_out = rgb_model(rgb_tensor)
    rgb_prob = F.softmax(rgb_out, dim=1)[0].detach().numpy()

    # Frequency branch
    freq_img_pil = convert_to_frequency(image_pil)
    freq_tensor = freq_transform(freq_img_pil).unsqueeze(0)
    freq_out = freq_model(freq_tensor)
    freq_prob = F.softmax(freq_out, dim=1)[0].detach().numpy()

    # Ensemble prediction
    final_prob = (rgb_prob + freq_prob) / 2
    cls = np.argmax(final_prob)   # 0 = FAKE, 1 = REAL

    return cls, freq_img_pil


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🔍 A Hybrid Framework for Efficient and Robust Deepfake Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=350)

    if st.button("Predict"):

        # ---- FACE CHECK ----
        if not detect_face(img):
            st.error("❌ No face detected. Please upload a clear image with a face.")
            st.stop()

        with st.spinner("Analyzing image..."):
            label, freq_image = predict_image(img)

        # ---- Final Output (NO CONFIDENCE) ----
        if label == 1:
            st.subheader("🟢 REAL")
        else:
            st.subheader("🔴 FAKE")

        # Show FFT image only
        st.write("### Frequency Domain Image (FFT)")
        st.image(freq_image, width=350)
