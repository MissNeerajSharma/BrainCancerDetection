# 🧠 AI-Powered Brain CT Scan Tumor Detection

A deep learning–based Flask web application to detect and classify brain tumors from CT scans into four categories:
- **Glioma**
- **Meningioma**
- **Pituitary**
- **No Tumor**

This project leverages **VGG16**, a pre-trained CNN model trained on over 14 million images, and fine-tuned using a private, doctor-annotated dataset. The system assists radiologists and doctors in early and accurate brain tumor diagnosis.

---

## 📸 Preview

![App Screenshot](https://github.com/MissNeerajSharma/BrainCancerDetection/blob/main/Screenshot%202025-06-02%20145129.png) <!-- Add your own screenshot -->

---

## 🚀 Demo

🧪 Try uploading a brain CT image and get predictions in real time.  
The model returns the **tumor type** and **prediction confidence score**.

---

## 📂 Dataset (Private)

- Annotated and verified by certified doctors.
- 4 categories, ~1500 images per class.
- Due to government restrictions, the dataset is not publicly available.

---

## 🧠 Model: VGG16

- Transfer learning with fine-tuning.
- Input shape: 128x128x3.
- Optimizer: Adam.
- Loss: Categorical Crossentropy.
- Accuracy achieved: _~XX%_ (fill actual result from training logs).

---

## 🔧 Tech Stack

| Tool | Purpose |
|------|---------|
| **Python** | Core programming language |
| **TensorFlow / Keras** | Deep learning and model training |
| **Flask** | Backend and web API |
| **HTML/CSS** | Frontend interface |
| **Pillow** | Image preprocessing |
| **Jinja2** | HTML templating |

---

## 🛠️ How to Run Locally

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/brain-tumor-detector.git
cd brain-tumor-detector
