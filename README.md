# Hybrid Deepfake Detection System

A hybrid deepfake detection system built using **PyTorch and EfficientNet-B4** that analyzes both **RGB images** and their **frequency domain (FFT) representations** to improve detection accuracy.

The system uses a **dual-model ensemble approach** and provides an **interactive Streamlit interface** for uploading images and predicting whether they are **REAL or FAKE**.

## Features

* Deepfake detection using **EfficientNet-B4**
* Hybrid approach using **RGB + Frequency (FFT) features**
* **Face detection** before prediction
* **Streamlit web interface** for easy image upload and prediction

## Tech Stack

* Python
* PyTorch
* EfficientNet-B4
* OpenCV
* Streamlit
* NumPy & PIL

## Project Structure

```
Hybrid-Deepfake-Detection-System
│
├── app.py
├── model/
│   ├── efficientnet_b4.py
│   └── efficientnet_b4_gray.py
├── RGB.pth
├── frequency.pth
└── README.md
```

## Installation

1. Clone the repository

```
git clone https://github.com/Navanith-Krishna-R/Hybrid-Deepfake-Detection-System.git
```

2. Install dependencies

```
pip install torch torchvision streamlit pillow numpy opencv-python efficientnet_pytorch
```

3. Run the application

```
streamlit run app.py
```

## Usage

Upload an image containing a face and click **Predict**.
The system will analyze the image and classify it as **REAL** or **FAKE**.

## Note

Model weights (`RGB.pth` and `frequency.pth`) may not be included due to GitHub file size limits. Place them in the project root directory before running the application.

## Author

Navanith Krishna R
