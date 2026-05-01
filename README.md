# 🔬 HematoScan AI — Blood Cancer Detection System

An AI-powered web application that detects **Acute Lymphoblastic Leukemia (ALL)** from blood smear microscopy images using a deep learning CNN model.

> ⚠️ This project is for **educational and research purposes only**. Not a medical diagnostic tool.

---

## 🚀 Features

* 🧠 Deep Learning based cancer detection (CNN)
* 🖼️ Upload blood smear images for instant analysis
* 📊 Confidence score + risk level (LOW / MODERATE / HIGH)
* 🔬 Cell feature extraction (density, chromatin, etc.)
* 🎨 Visual analysis overlay on image
* 📁 Batch image prediction support
* 🌐 Clean and modern UI (drag & drop)

---

## 🏗️ Project Structure

```
blood-cancer-app/
├── app.py                    # Flask backend (API + inference)
├── run.py                    # App launcher
├── requirements.txt         # Dependencies
├── templates/
│   └── index.html           # Frontend UI
├── model/
│   ├── train_model.py       # Training script
│   ├── model_meta.json      # Model metadata
│   └── blood_cancer_model.keras (ignored)
└── uploads/                 # Temporary files
```

---

## 🤖 AI Model Details

* **Architecture**: Custom CNN (6 Conv Blocks)
* **Input Size**: 96 × 96 × 3
* **Output**: Binary Classification (Benign vs Malignant)
* **Framework**: TensorFlow / Keras
* **Loss Function**: Binary Crossentropy
* **Optimizer**: Adam

### 📊 Classes

* `0 → Benign (Normal HEM)`
* `1 → Malignant (ALL Blast Cells)`

---

## 📁 Dataset

* **Source**: Kaggle Leukemia Dataset
* **Classes**: Healthy (HEM) vs Leukemia (ALL)
* **Images**: 4000+ microscopy images

👉 https://www.kaggle.com/datasets/andrewmvd/leukemia-classification

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/blood-cancer-detection.git
cd blood-cancer-detection
```

---

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Run the application

```bash
python run.py
```

---

### 4️⃣ Open in browser

```
http://localhost:5000
```

---

## 📡 API Endpoint

### POST `/predict`

Upload an image for prediction.

#### Request:

* `multipart/form-data`
* field name: `image`

#### Response:

```json
{
  "prediction": {
    "label": "Malignant",
    "confidence": 92.5,
    "risk_level": "HIGH"
  },
  "cell_features": {
    "nuclear_density": 45.2,
    "chromatin_intensity": 60.3
  }
}
```

---

## 🧪 How It Works

1. User uploads blood smear image
2. Image is preprocessed (resize + normalize)
3. CNN model predicts cancer probability
4. Features are extracted from image
5. Results are returned with visualization

---

## 📸 Screenshots

> Add screenshots here after uploading images

---

## ⚠️ Disclaimer

* This system is **NOT a clinical diagnostic tool**
* Predictions may not be accurate for real-world medical use
* Always consult a qualified doctor or pathologist

---

## 🤝 Contributing

Pull requests are welcome!
For major changes, please open an issue first.

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 👨‍💻 Author

**Akshat Tiwari**

---

## ⭐ Support

If you found this project useful:

👉 Give it a star ⭐ on GitHub
👉 Share with your friends

---
