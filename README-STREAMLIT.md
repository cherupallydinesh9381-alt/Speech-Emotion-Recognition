# 🎤 DeepSER-GUI - Speech Emotion Recognition

**Professional Streamlit Web Application for Real-time Emotion Detection**

[![Streamlit](https://img.shields.io/badge/Streamlit-1.52-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.11-FF6F00?style=for-the-badge&logo=tensorflow)](https://tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python)](https://python.org/)

---

## 📋 Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Model Information](#model-information)
- [Deployment](#deployment)
- [Screenshots](#screenshots)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## ✨ Features

### 🎯 Core Features
- ✅ **Real-time Emotion Detection** - Analyze emotions in <500ms
- ✅ **Multi-format Support** - WAV, MP3, FLAC, OGG
- ✅ **Multiple Emotion Sets** - 3, 5, or 9 emotion classes
- ✅ **Visual Analysis** - Waveform & Spectrogram display
- ✅ **Confidence Scores** - Probability distribution charts
- ✅ **Pre-trained Models** - Ready-to-use LSTM models

### 🎨 UI Features
- Modern, responsive design
- Interactive visualizations
- Real-time audio playback
- Drag-and-drop file upload
- Mobile-friendly interface

### 🧠 Supported Emotions

**3 Emotions (HNS):** Happy, Neutral, Sad

**5 Emotions (AHNPS):** Angry, Happy, Neutral, Pleasant Surprise, Sad

**9 Emotions (All):** Neutral, Calm, Happy, Sad, Angry, Fear, Disgust, Pleasant Surprise, Boredom

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip package manager

### Install & Run

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd Speech-Emotion-Recognition-Using-Deep-CNN-master

# 2. Install dependencies
pip install -r requirements-streamlit.txt

# 3. Run the application
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

---

## 📦 Installation

### Option 1: Standard Installation

```bash
# Install all dependencies
pip install -r requirements-streamlit.txt
```

### Option 2: Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements-streamlit.txt
```

### Option 3: Conda Environment

```bash
# Create conda environment
conda create -n deepser python=3.10

# Activate environment
conda activate deepser

# Install dependencies
pip install -r requirements-streamlit.txt
```

---

## 🎮 Usage

### Step 1: Launch Application

```bash
streamlit run app.py
```

### Step 2: Configure Settings

1. **Select Emotion Set** (Sidebar)
   - Choose: 3, 5, or 9 emotions
   
2. **Visualization Options**
   - Toggle waveform display
   - Toggle spectrogram display
   - Toggle probability charts

### Step 3: Upload & Analyze

1. **Upload Audio File**
   - Click "Browse files" or drag & drop
   - Supported formats: WAV, MP3, FLAC, OGG
   - Maximum size: 20 MB

2. **Listen to Audio**
   - Use built-in audio player

3. **Analyze Emotion**
   - Click "Analyze Emotion" button
   - View results in ~500ms

### Step 4: Interpret Results

- **Primary Emotion:** Large display with emoji
- **Confidence Score:** Percentage of certainty
- **Probability Chart:** All emotions ranked
- **Audio Visualizations:** Waveform & spectrogram
- **Detailed Metrics:** Individual emotion scores

---

## 🤖 Model Information

### Architecture

```
Deep Recurrent Neural Network (LSTM)
├── Input Layer: 180 features (MFCC + Chroma + MEL)
├── LSTM Layer 1: 128 units
├── Dropout: 0.3
├── LSTM Layer 2: 128 units
├── Dropout: 0.3
├── Dense Layer 1: 128 units
├── Dropout: 0.3
├── Dense Layer 2: 128 units
├── Dropout: 0.3
└── Output Layer: Softmax (3/5/9 classes)
```

### Training Data

- **RAVDESS:** Ryerson Audio-Visual Database (24 actors)
- **TESS:** Toronto Emotional Speech Set (200 words)
- **EMO-DB:** Berlin Database of Emotional Speech

### Features Extracted

1. **MFCC** (40 coefficients) - Voice quality
2. **Chroma** (12 features) - Pitch characteristics
3. **MEL Spectrogram** (128 bins) - Frequency distribution

### Performance

| Model | Emotions | Test Accuracy |
|-------|----------|---------------|
| HNS   | 3        | ~36.5%        |
| AHNPS | 5        | ~20.6%        |

*Note: Accuracy varies based on audio quality and dataset*

---

## 🌐 Deployment

### Local Deployment

```bash
streamlit run app.py --server.port 8501
```

### Streamlit Cloud

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Add Streamlit app"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io/)
   - Connect GitHub repository
   - Select `app.py` as main file
   - Deploy!

3. **Configuration** (optional)
   
   Create `.streamlit/config.toml`:
   ```toml
   [theme]
   primaryColor = "#1E88E5"
   backgroundColor = "#FFFFFF"
   secondaryBackgroundColor = "#F0F2F6"
   textColor = "#262730"
   font = "sans serif"
   
   [server]
   maxUploadSize = 20
   ```

### Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements-streamlit.txt .
RUN pip install --no-cache-dir -r requirements-streamlit.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

Build & Run:
```bash
docker build -t deepser-gui .
docker run -p 8501:8501 deepser-gui
```

---

## 📸 Screenshots

### Main Interface
![Main Interface](images/screenshot_main.png)

### Emotion Detection Results
![Results](images/screenshot_results.png)

### Audio Visualization
![Visualization](images/screenshot_visualization.png)

---

## 📁 Project Structure

```
Speech-Emotion-Recognition-Using-Deep-CNN-master/
├── app.py                          # Main Streamlit application
├── deep_emotion_recognition.py     # Deep learning model class
├── emotion_recognition.py          # Traditional ML model class
├── utils.py                        # Utility functions
├── data_extractor.py              # Data loading utilities
├── create_csv.py                  # CSV generation
├── requirements-streamlit.txt      # Streamlit dependencies
├── requirements.txt               # Original dependencies
├── results/                       # Pre-trained models
│   ├── HNS-c-LSTM-*.h5           # 3 emotion model
│   ├── AHNPS-c-LSTM-*.h5         # 5 emotion model
│   └── HNS-r-LSTM-*.h5           # Regression model
├── features/                      # Pre-extracted features
│   ├── train_mfcc-*.npy
│   └── test_mfcc-*.npy
├── images/                        # UI assets
└── README-STREAMLIT.md           # This file
```

---

## 🔧 Troubleshooting

### Issue: Module not found

```bash
# Solution: Reinstall dependencies
pip install -r requirements-streamlit.txt
```

### Issue: Model not loading

```bash
# Solution: Verify model files exist
ls results/*.h5

# If missing, models will initialize (lower accuracy)
```

### Issue: Audio upload fails

**Check:**
- File size < 20 MB
- Format: WAV, MP3, FLAC, or OGG
- File not corrupted

### Issue: Slow performance

**Solutions:**
- Use smaller audio files (< 10 seconds)
- Reduce visualization options
- Close other applications
- Use GPU-enabled TensorFlow

### Issue: Incorrect predictions

**Common causes:**
- Low audio quality
- Background noise
- Multiple speakers
- Ambiguous emotion

**Solutions:**
- Use clear, single-speaker audio
- Record in quiet environment
- Ensure emotional expression is clear

---

## 🎓 Academic Use

This project is designed for:
- **Master's Thesis/Project**
- **Research Papers**
- **Educational Demonstrations**
- **Emotion Recognition Studies**

### Citation

If you use this work, please cite:

```bibtex
@software{deepser_gui,
  title = {DeepSER-GUI: Speech Emotion Recognition using Deep CNN},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/deepser}
}
```

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Datasets:** RAVDESS, TESS, EMO-DB
- **Framework:** TensorFlow/Keras, Streamlit
- **Libraries:** librosa, scikit-learn, matplotlib

---

## 📞 Support

For issues, questions, or suggestions:

- **GitHub Issues:** [Create an issue](https://github.com/yourusername/deepser/issues)
- **Email:** your.email@example.com
- **Documentation:** [Wiki](https://github.com/yourusername/deepser/wiki)

---

## 🎯 Roadmap

- [ ] Add microphone recording feature
- [ ] Support batch processing
- [ ] Add model training interface
- [ ] Implement voice activity detection
- [ ] Add multi-language support
- [ ] Export analysis reports (PDF/CSV)
- [ ] Real-time streaming analysis
- [ ] Model comparison tool

---

<div align="center">

### 🌟 Star this repo if you find it helpful!

**Built with ❤️ using Streamlit & TensorFlow**

[⬆ Back to Top](#-deepser-gui---speech-emotion-recognition)

</div>
