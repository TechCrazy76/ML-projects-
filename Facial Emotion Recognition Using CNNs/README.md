# Facial Emotion Recognition Using CNNs | Computer Vision |
[Feb 2025 - Apr 2025]

(Self Project)

◦ Trained a CNN in TensorFlow/Keras on the FER-2013 dataset with face alignment and preprocessing
using OpenCV, achieving 66% accuracy with full evaluation via precision recall, F1 and confusion matrix

◦ Deployed a real-time emotion detection system using Haar cascades, enabling live webcam inference across
7 emotions with probability overlays

---

## 🔍 Overview

A compact, end-to-end pipeline for **Facial Emotion Recognition** built with TensorFlow 2 / Keras and OpenCV.  
Includes preprocessing (face alignment), CNN training on the **FER-2013 dataset**, model evaluation, and a real-time webcam demo using Haar cascades for face detection.

---

## 🧠 Key Achievements
- Trained a **CNN on FER-2013** (preprocessed & face-aligned) achieving **~66–67% test accuracy**.  
- Generated detailed **classification reports, confusion matrices, and per-class metrics**.  
- Implemented a **real-time webcam demo** using Haar cascades + probability overlays for 7 emotions:  
  `angry, disgusted, fearful, happy, sad, surprised, neutral`.  
- Saved model (`.h5`) and processed datasets (`.npy`) for reproducibility.  

---

## 🔍 Project Overview

### **Preprocessing**
- Input: `fer2013.csv` (from Kaggle).  
- Converted pixel strings to images, detected faces with **OpenCV Haar cascades**, cropped/resized to `48×48`, and normalized `[0,1]`.  
- Saved processed datasets: data/sf=1.1_data_images.npy, data/sf=1.1_data_labels.npy

### **Training**
- CNN architecture: `Conv2D → MaxPool → Conv2D → Dropout → Dense`.  
- Loss: **MSE**, Optimizer: **Adam**.  
- Utilities: `ReduceLROnPlateau`, `EarlyStopping`, `ModelCheckpoint`.  
- Example model: `Gudi_model_100_epochs_20000_faces_keras.h5`.

### **Evaluation**
- Computed **Precision, Recall, F1-score** per class.  
- Generated **Confusion Matrix & Normalized Confusion Matrix** visualizations.  
- Saved performance plots and reports.

### **Inference**
- `predict_keras.py` – single image inference (with bounding boxes & probabilities).  
- `predict_cam_video_keras.py` – **live webcam demo** (local mode).  
- `predict_mtcnn.py` – alternative detector (optional).

### **Deployment**
- **Local:** Run webcam demo via:
```bash
python predict_cam_video_keras.py --model data/Gudi_model_100_epochs_20000_faces_keras.h5
```
◦ **Colab**: Fully reproducible pipeline with setup cells for preprocessing, training, and evaluation.

---

## 🔢 Final Quantitative Results

| **Metric** | **Value** |
|:------------|:----------|
| **Test Accuracy (FER-2013 processed)** | **~66–67%** |
| **Macro Avg F1** | ~0.62–0.65 |
| **Weighted Avg F1** | ~0.66–0.67 |

---

### 🎯 Per-Class Performance Highlights

| **Emotion** | **F1-Score** | **Notes** |
|:-------------|:-------------|:-----------|
| 😄 **Happy** | 0.84–0.86 | Strong performance |
| 😲 **Surprised** | 0.76–0.80 | Good accuracy |
| 😢 **Sad / Disgusted** | Lower recall | Often confused with *neutral* |
| 😐 **Neutral / Angry** | Moderate | Some overlap |

---

## 📁 Repository Structure

```powershell
.
├─ constants.py
├─ data_process.py
├─ dataset_loader.py
├─ train_keras.py
├─ predict_keras.py
├─ predict_cam_video_keras.py
├─ predict_mtcnn.py
│
├─ pics/                # Example input images
├─ haarcascades/        # Haar cascade XMLs
├─ emojis/              # Optional overlay icons
│
├─ data/
│  ├─ fer2013.csv       # Kaggle download
│  ├─ sf=1.1_data_images.npy
│  ├─ sf=1.1_data_labels.npy
│  └─ Gudi_model_100_epochs_20000_faces_keras.h5
│
└─ README.md

---

## 🧩 Requirements

- **Python 3.8+**
- **TensorFlow 2.x / Keras**
- **OpenCV**
- **NumPy**, **Pandas**, **Pillow**, **Scikit-learn**, **Matplotlib**, **Seaborn**

### Install:

```bash
pip install -r requirements.txt
```
# or *individually*
pip install numpy pandas opencv-python-headless pillow scikit-learn tensorflow keras h5py matplotlib seaborn

---

## 🚀 Quick Start

### 🧠 Google Colab (Recommended for GPU)

1. **Change runtime to GPU (T4)**  
   - Navigate to: `Runtime → Change runtime type → Hardware accelerator → GPU`.

2. **Run setup cell:**
```bash
!pip install -q numpy pandas opencv-python-headless pillow scikit-learn tensorflow keras h5py tflearn
```
3. **Upload assets** (data.zip, haarcascades.zip, pics.zip) in Colab Files.

4. **Run all preprocessing → training → evaluation cells in order.**

Example:
```bash
!python predict_keras.py --image pics/1happy.jpg
```
---

## 💻 Local (Windows / Linux / macOS)
```bash
git clone <repo_url>
cd <repo_name>
python -m venv venv
# Activate venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # macOS/Linux
pip install -r requirements.txt
```
Place **fer2013.csv** inside **data/**, then run:
```bash
python data_process.py
python train_keras.py
python predict_keras.py --image pics/1happy.jpg
python predict_cam_video_keras.py --model data/Gudi_model_100_epochs_20000_faces_keras.h5
```
Press q to quit webcam window.

---

## ✨ Usage Examples

### Single Image Prediction
```bash
python predict_keras.py --image pics/1happy.jpg --model data/Gudi_model_100_epochs_20000_faces_keras.h5
```

### Live Webcam (Local Only)
```bash
python predict_cam_video_keras.py --model data/Gudi_model_100_epochs_20000_faces_keras.h5
```

---

## 📈 Evaluation Summary & Insights

1. Model achieves **≈66–67% test accuracy** on FER-2013.
2. **Happy** and **Surprised** detected most reliably.
3. **Sad**, **Disgusted**, and **Neutral** show frequent confusion (common in FER datasets).
4. Overall performance aligns with baseline FER benchmarks for compact CNNs.

---

## ✅ Suggested Improvements

1. **Data Augmentation:** Random flips, rotations, brightness changes.
2. **Class Balancing:** Use weighted loss or focal loss.
3. **Better Face Alignment:** Try dlib or MTCNN for landmark-based preprocessing.
4. **Stronger Models:** Experiment with ResNet, MobileNet, or transfer learning.
5. **Model Ensembling:** Combine CNNs with different seeds or architectures.
6. **Dataset Expansion:** Merge with RAF or other FER datasets.

---

## ❗ Licensing & Data Note

1. FER-2013 dataset must be **downloaded from Kaggle** due to licensing.
2. Place **fer2013.csv** inside **/data** before preprocessing.
3. Pretrained model and .npy datasets are provided for demo purposes only.

---

## 🔗 External Assets

1. FER-2013: [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
2. Google Drive link for model and .npy datasets: [drive](https://drive.google.com/drive/folders/1GK-uREganN9yG59kESzYwpuCFM64HE1h?usp=drive_link)

---

## 📸 Sample Output

![Confusion Matrix](Confusion matrix.jpeg)

![Classification Report](Normalised confusion matrix.jpeg)

![Sample output picture](3surprised.jpg_out.png)

---













