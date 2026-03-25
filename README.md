# power-quality-event-classification
Machine learning based classification of power quality disturbances using DWT and multi-model evaluation
# ⚡ Power Quality Event Classification using Machine Learning

## 📌 Overview
This project presents an intelligent system for **classification of power quality (PQ) disturbances** using signal processing and machine learning techniques.

The system analyzes voltage signals, extracts meaningful features using **Discrete Wavelet Transform (DWT)**, and classifies disturbances using multiple machine learning models.


## 🎯 Objectives
- Detect and classify different power quality disturbances
- Ensure robustness under noisy conditions
- Compare performance of multiple ML models
- Simulate real-world power system scenarios

## ⚙️ Power Quality Events Considered
- Normal
- Voltage Sag
- Voltage Swell
- Interruption
- Harmonics
- Transient
- Flicker
- Notching

## 🧠 Methodology

### 1. Signal Generation
- Synthetic signals generated based on IEEE 1159 standard models

### 2. Preprocessing
- Normalization
- Noise addition (20–50 dB SNR)
- High-pass filtering (DC removal)

### 3. Feature Extraction
- Discrete Wavelet Transform (db4, 5 levels)
- Extracted 36 statistical features:
  - Energy
  - Entropy
  - RMS
  - Kurtosis
  - THD
  - Peak values

### 4. Classification Models
- Support Vector Machine (SVM)
- Random Forest
- Boosted Trees
- Neural Network

## 📊 Results

### 🔥 Accuracy (SNR = 40 dB)
| Model            | Accuracy |
|-----------------|----------|
| SVM             | 97.17%   |
| Random Forest   | 98.67%   |
| Boosted Trees   | **98.83% (Best)** |
| Neural Network  | 97.33%   |

### 📈 Key Highlights
- High classification accuracy (~99%)
- Robust performance under noisy conditions
- Effective feature extraction using DWT
- Boosted Trees achieved best performance

## 📉 Noise Robustness
The system was tested under different noise levels:
- 50 dB (clean)
- 40 dB
- 30 dB
- 20 dB (heavy noise)

👉 The model maintains strong performance even under heavy noise conditions.

## 📂 Project Structure


power-quality-event-classification/
│
├── main.py
├── README.md
├── requirements.txt
│
├── results/
│ ├── accuracy.png
│ ├── confusion_matrix.png
│ ├── f1_scores.png
│ ├── feature_importance.png
│ ├── preprocessing.png
│ ├── dwt.png
│ ├── robustness.png
│
├── images/
│ ├── waveforms.png
│ ├── heatmap.png

## 📊 Sample Outputs

### 🔹 Preprocessing Pipeline
![Preprocessing](results/preprocessing.png)

### 🔹 DWT Decomposition
![DWT](results/dwt.png)

### 🔹 Classification Accuracy
![Accuracy](results/accuracy.png)

### 🔹 Confusion Matrix
![Confusion Matrix](results/confusion_matrix.png)

### 🔹 Feature Importance
![Feature Importance](results/feature_importance.png)

### 🔹 Noise Robustness
![Robustness](results/robustness.png)

## 🛠️ Technologies Used
- Python
- NumPy
- SciPy
- Scikit-learn
- PyWavelets
- Matplotlib

## 🚀 Future Scope
- Real-time implementation using embedded systems
- Integration with smart grid monitoring
- Deployment using IoT and cloud platforms

## 👨‍💻 Author
Prem Santhosh Kumar Lomada
## ⭐ Conclusion
This project demonstrates a **highly accurate and robust system** for 
