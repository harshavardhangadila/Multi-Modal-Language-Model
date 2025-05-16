# Multi-Modal-Language-Model


This repository demonstrates multiple AI/ML tasks across different data modalities—image, text, audio, video, tabular, and time series—using tools like PyTorch, TensorFlow, OpenCV, Hugging Face Transformers, Scikit-learn, Statsmodels, and Torchaudio.

---

## 📁 Contents

### 1. 🔤 Zero-Shot Image Classification (CLIP)
- Uses Hugging Face `CLIPModel` and `CLIPProcessor`
- Performs classification on unseen images using natural language labels
- GPU acceleration supported

### 2. 📊 Tabular Regression (Diabetes Dataset)
- Linear Regression using Scikit-learn
- Metrics: MSE, RMSE, R² Score
- Visualizes actual vs predicted values

### 3. 📈 Tabular Clustering (Wine Dataset)
- Uses KMeans clustering
- Evaluates with Elbow Method, Silhouette Score, ARI, and NMI
- PCA-based cluster visualization

### 4. 🧠 Text Classification (Sentiment Analysis)
- Dataset: `nltk.corpus.movie_reviews`
- TF-IDF vectorization + Logistic Regression
- Includes confidence scores and sample predictions

### 5. 🖼️ Image Classification (CIFAR-10 with PyTorch)
- Custom CNN using `torch.nn`
- Trained on `torchvision.datasets.CIFAR10`
- Per-class accuracy and prediction display

### 6. 📉 Time Series Forecasting (Mauna Loa CO₂)
- Dataset: `statsmodels.datasets.co2`
- ARIMA forecasting with trend/seasonality decomposition
- Monthly interpolation, visual diagnostics

### 7. 🔊 Audio Signal Processing (YESNO Dataset)
- Loads and visualizes waveform and mel spectrogram
- Dataset: `torchaudio.datasets.YESNO`
- Interactive playback of samples (Jupyter/Colab)

### 8. 🔢 Image Classification (MNIST with TensorFlow)
- CNN built using `tf.keras.Sequential`
- Dataset: `tensorflow_datasets.mnist`
- GPU setup, accuracy/loss plots, and prediction examples

## 9. 🎥 Video Classification: Action Recognition with R3D-18

- Loads and decodes `.mp4` videos using OpenCV.
- Extracts RGB frames and preprocesses them into `[1, C, T, H, W]` format.
- Uses the pretrained `r3d_18` 3D CNN from `torchvision.models.video`.
- Performs inference using Kinetics-400 action label set.
- Applies softmax to produce probabilities for each action.
- Displays the top-5 predicted actions with confidence scores.
- Visualizes the center frame of the sampled video clip.

---

Youtube: [Multi-Modal-Language-Model-Use-Cases](https://www.youtube.com/playlist?list=PLCGwaUpxPWO1BieFZsw6ulGSFzwhv3g1v)
