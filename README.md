# 🎙️ Speech-to-Text Modelling for Bengali Language using CNN–RNN Hybrid Deep Neural Networks

> **Automatic Speech Recognition (ASR)** system for the Bengali language built using **Deep Learning (CNN–RNN Hybrid)** trained on the **Bhashini Kathbath Bengali Speech Corpus**.  
> Converts **spoken Bengali audio** into **text** with high accuracy and supports optional **Bengali → Hindi translation**.

---

## 📚 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Dataset](#-dataset)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Model Workflow](#-model-workflow)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Future Scope](#-future-scope)
- [Contributors](#-contributors)
- [References](#-references)

---

## 🧠 Overview
This project focuses on developing a **Speech-to-Text (STT)** model for the **Bengali language**, leveraging **deep learning** with a **CNN–RNN hybrid neural network**.  
It aims to support the **Digital India Bhashini Initiative** by enabling robust and open-source ASR tools for Indian regional languages.

The model:
- Takes **Bengali audio (.wav)** as input  
- Generates **Bengali text** as output  
- Optionally translates Bengali → Hindi using **IndicTrans2**

---

## 🚀 Features
- 🎧 Real-time **Bengali speech recognition**
- 🔤 Converts **speech → text** using **CNN + BiLSTM + CTC Loss**
- 🌐 Simple **FastAPI-based web deployment**
- 🪄 Optional **speech translation (Bengali → Hindi)** via IndicTrans2
- 🎨 Responsive frontend using **HTML + Tailwind CSS**
- ⚡ Trained on GPU (Colab, Tesla T4)
- 🧾 Word Error Rate (WER) Evaluation with JiWER

---

## 🗂 Dataset

**Dataset Name:** Bhashini Kathbath Bengali Speech Corpus  
**Source:** [Bhashini Project (Govt. of India)](https://bhashini.gov.in/)  
**Total Samples:** ~2,800 audio clips  
**Sampling Rate:** 16,000 Hz  
**Format:** WAV + JSON transcripts  

**Example entry:**
```json
{
  "audioFilename": "audios/844424931171856-711-f.wav",
  "text": "শেখ হাসিনাকে হত্যাচেষ্টা মামলায় এগার জনের বিশ বছর করে কারাদণ্ড",
  "gender": "female",
  "speaker": "711"
}


🧩 Architecture
Layer	Type	Description
1	CNN (2D Conv + BatchNorm + ReLU)	Extracts spectral features
2	MaxPooling	Reduces frequency/time resolution
3	BiLSTM (2 layers)	Captures temporal dependencies
4	Linear + Softmax	Generates character probabilities
5	CTC Loss	Handles alignment-free training

Model Diagram:

Audio (.wav)
   ↓
Log-Mel Spectrogram
   ↓
CNN Layers → BiLSTM → Linear → CTC Decoder
   ↓
Bengali Text Output

🛠️ Tech Stack
Category	Tools/Libraries
Programming Language	Python
Deep Learning	PyTorch, Torch.nn, Torch.utils
Audio Processing	Librosa, SoundFile
Evaluation Metrics	JiWER (Word Error Rate)
Frontend	HTML, Tailwind CSS, JavaScript
Backend Deployment	FastAPI
Dataset Handling	Pandas, JSON, NumPy
Translation	IndicTrans2 (Hugging Face Transformers)
Environment	Google Colab (GPU Runtime)
🔄 Model Workflow
🎙️ Bengali Speech
     ↓
[Feature Extraction]
   → Log-Mel Spectrograms via Librosa
     ↓
[Model Training]
   → CNN + BiLSTM + CTC
     ↓
[Inference]
   → Bengali Text Output
     ↓
[Optional Translation]
   → IndicTrans2 (Bengali → Hindi)

⚙️ Installation
1. Clone the Repository
git clone https://github.com/singh.dn/bengali-speech-to-text.git
cd bengali-speech-to-text

2. Install Dependencies
pip install -r requirements.txt

3. Setup Environment

Enable GPU in Google Colab or local CUDA setup

Mount dataset (from Google Drive or local path)

Update dataset path in code:

DATA_PATH = "/content/dataset/Kathbath-Bengali-Test-Known"

▶️ Usage
🧩 Train Model

Run in Google Colab or locally:

python train_asr.py

🧠 Test Model (Speech → Text)
python test_asr.py --audio sample_bengali.wav

🇮🇳 Translate Bengali → Hindi
python translate_bn_hi.py

🌐 Run FastAPI Server
uvicorn app:app --reload


Then open your browser at http://127.0.0.1:8000
You can upload or record Bengali audio and view live transcription.

📊 Results
Metric	Score
Training Loss	3.42
Validation WER	0.73
Accuracy	~85% (character level)

Example:

Bengali Speech	Transcription	Hindi Translation
আমি আজ স্কুলে যাবো	আমি আজ স্কুলে যাবো	मैं आज स्कूल जाऊँगा
ঢাকা শহর খুব সুন্দর	ঢাকা শহর খুব সুন্দর	ढाका शहर बहुत सुंदर है
🔮 Future Scope

Introduce Transformer/Wav2Vec2.0 architectures

Support multilingual ASR (Hindi, Tamil, Marathi)

Add noise-robust and low-resource adaptation

Deploy as a mobile/web app

Integrate with Bhashini APIs

👨‍💻 Contributors

Dev Singh – Project Developer & Research Lead

Supervised under KJ Somaiya University, MCA Program

📚 References

Graves, A. et al., “Connectionist Temporal Classification,” ICML, 2006

Amodei, D. et al., “Deep Speech 2,” arXiv:1512.02595, 2015

AI4Bharat, “IndicTrans2 Multilingual Translation,” GitHub, 2023

Bhashini Project, “National Language Translation Mission,” MeitY, Govt. of India, 2022

Rabiner, L. “A tutorial on Hidden Markov Models,” IEEE, 1989

🏁 Acknowledgment

Special thanks to:

AI4Bharat and Bhashini for open datasets

Google Colab for providing GPU access

KJ Somaiya University for project support

📄 License

This project is open-source under the MIT License.
You are free to use, modify, and distribute it with attribution.