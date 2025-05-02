# ML Image Detection - Civic Issue Classifier

This project uses a trained MobileNetV2-based deep learning model to classify civic infrastructure issues from images. The model is capable of detecting three categories:
- **Potholes**
- **Open Sewers**
- **Others**

It is built using TensorFlow 2.10 and runs on Python 3.9.0.

---

## 🗂️ Project Structure

```
.
├── best_model.h5               # Trained MobileNetV2 model
├── run_model.py                # Python script to run inference
├── requirements.txt            # Python dependencies
├── TestImages/                 # Folder containing test images
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── README.md                   # Project documentation
```

---

## ⚙️ Setup Instructions

### 1. ✅ Install Python 3.9.0

Ensure you're using Python 3.9.0. You can check this using:
```bash
python --version
```

### 2. ✅ Create and activate a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. ✅ Install required packages

Install all dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## ⚡ (Optional) GPU Acceleration Setup

> 💡 This step is **optional**. The project works on CPU by default.  
> Setting up GPU acceleration improves performance for large-scale inference.

To run TensorFlow on GPU, ensure the following are installed and configured:

### ✅ Required Versions for TensorFlow 2.10:

- **CUDA Toolkit**: 11.2  
- **cuDNN**: 8.1.1  
- **NVIDIA GPU Driver**: Compatible with CUDA 11.2  
- **GPU**: CUDA-enabled (e.g., RTX 3050 or similar)

---

### 🧰 GPU Setup Steps (Windows)

#### 1. Install CUDA Toolkit 11.2

Download and install from the [NVIDIA CUDA 11.2 Archive](https://developer.nvidia.com/cuda-11.2.0-download-archive).

By default, it installs to:
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\
```

---

#### 2. Install cuDNN 8.1.1

- Download cuDNN 8.1.1 for Windows from the [NVIDIA cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive).
- Extract the ZIP file.
- Inside the extracted folder (e.g., `cudnn-windows-x86_64-8.1.1.33_cuda11.2`), you’ll find 3 folders:
  - `bin`
  - `include`
  - `lib`

👉 **Copy the contents of each to your CUDA installation:**

| From cuDNN Folder | To CUDA Folder |
|------------------|----------------|
| `bin\cudnn64_8.dll` | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin\` |
| `include\cudnn*.h`  | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include\` |
| `lib\x64\cudnn*.lib`| `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\lib\x64\` |

✅ Overwrite any existing files when prompted.

---

#### 3. Add CUDA to Environment Variables

Add the following paths to your system’s `PATH` variable:

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp
```

---

#### 4. Verify GPU Detection

Run the following in Python:

```python
import tensorflow as tf
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
```

Expected output should detect your GPU:
```
Num GPUs Available: 1
```

---

## 🚀 Running the Model

1. Place your test images in the `TestImages/` folder.

2. Run the inference script:

```bash
python run_model.py
```

3. The script will:
   - Load the trained model (`best_model.h5`)
   - Preprocess each image in `TestImages/`
   - Predict the class label and print the confidence
   - Optionally display each image using `matplotlib`

### 🖼️ Example Output

```
img1.jpg → Potholes (91.34% confidence)
img2.jpg → Open Sewers (88.27% confidence)
```

---

## 📋 Notes

- The model was trained using MobileNetV2 with transfer learning on a civic issue dataset.
- The `run_model.py` script automatically handles image preprocessing and prediction.
- GPU is automatically used if available; otherwise, it falls back to CPU.

---

## 💡 Troubleshooting

- If the model file is too large to upload to GitHub (>100MB), consider using [Git LFS](https://git-lfs.github.com/).
- For errors loading images, ensure they are in a supported format (`.jpg`, `.png`, etc.) and not corrupted.
- If TensorFlow isn't detecting your GPU, verify that installed CUDA and cuDNN versions match your TensorFlow version (2.10.0 → CUDA 11.2 + cuDNN 8.1.1).

---

## 👤 Author

**Mayukh Tilak (CSE Student - Cyber Security Specialization)**  
For academic use, demos, and research. Contributions welcome!

---
