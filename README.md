# 🩺 Swasthya — Offline Bilingual Medical Assistant

> An offline-first medical first-response assistant fine-tuned on a bilingual Bengali/English dataset, running on-device via a quantized GGUF model — built for low-resource environments and rural healthcare access.

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)](https://fastapi.tiangolo.com/)
[![LlamaCpp](https://img.shields.io/badge/llama--cpp--python-0.2.90-orange.svg)](https://github.com/abetlen/llama-cpp-python)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-yellow.svg)](https://huggingface.co/souravchandra01)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

---

## 🏗️ System Architecture

```
User Input (Bengali / English)
          ↓
  swasthya_ui.html
  (Dark UI + TTS)
          ↓
  FastAPI Backend
  (server.py)
          ↓
  Emergency Safety Check
  (keyword-based, bilingual)
          ↓
  llama-cpp-python
          ↓
  swasthya-1b-q4km.gguf
  (TigerLLM 1B — Q4_K_M)
          ↓
  Medical Response
  (Bengali or English)
```

---

## ✨ Key Features

- **🌐 Bilingual** - Responds in Bengali or English based on user input
- **📴 Offline-First** - Runs entirely without internet after setup
- **📱 Android Compatible** - Tested on Android via Termux + llama.cpp native binary
- **🚨 Emergency Safety Layer** - Detects critical symptoms and shows emergency warning in both languages
- **🔊 Text-to-Speech** - Built-in TTS using system speech engine (offline)
- **⚡ Quantized Inference** - Q4_K_M quantization for efficient CPU inference
- **🎯 Fine-tuned** - TigerLLM-1B-it fine-tuned on curated bilingual medical dataset

---

## 📹 Demo

**🎥 Demo video coming soon**

---

## 🛠️ Tech Stack

### Model
- **TigerLLM-1B-it** - Gemma 3 architecture, Bengali-optimized base model
- **LoRA (PEFT)** - Parameter-efficient fine-tuning
- **GGUF Q4_K_M** - 4-bit quantization (~700MB)
- **llama.cpp** - Efficient CPU/GPU inference engine

### Backend
- **FastAPI** - Lightweight API server
- **llama-cpp-python** - Python bindings for llama.cpp

### Frontend
- **Vanilla HTML/CSS/JS** - Single self-contained file, no dependencies
- **Web Speech API** - Offline TTS for Bengali and English

### Training
- **Google Colab** - Data processing and GGUF conversion
- **Kaggle (T4 GPU)** - Model fine-tuning
- **HuggingFace** - Model hosting

---

## 📊 Dataset

| Split | Source | Samples |
|---|---|---|
| English | HealthCareMagic-100k (filtered) | 1,000 |
| Bengali | Manually translated + curated | 500 |
| **Total** | | **1,500** |

**Filtering:** 112,165 raw samples → 2,567 cleaned → 1,500 used for training

---

## 🚀 Quick Start

### Requirements

```bash
cd app
pip install -r requirements.txt
```

> ⚠️ **Windows users:** If `llama-cpp-python` fails to install, use:
> ```bash
> pip install llama-cpp-python==0.2.90 --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
> ```

### Download Model

Download `swasthya-1b-q4km.gguf` from HuggingFace and place it in the `models/` folder:

🤗 [souravchandra01/swasthya-1b-gguf](https://huggingface.co/souravchandra01/swasthya-1b-gguf)

### Run

```bash
# Start the FastAPI server
python app/server.py

# Open the frontend
# Just open app/swasthya_ui.html in your browser
```

Server runs at `http://localhost:8000`

---

## 📱 Android Deployment (Termux)

The model was tested running natively on Android via Termux:

```bash
# Install llama.cpp in Termux
pkg install cmake clang

# Clone and build llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && cmake -B build && cmake --build build

# Run inference
./build/bin/llama-simple-chat \
  -m ~/storage/downloads/swasthya-1b-q4km.gguf
```

> Model size: ~700MB — tested on Android with ~3GB RAM available

---

## 🗂️ Project Structure

```
swasthya-medical-assistant/
│
├── notebooks/
│   ├── 01_data_filtering_colab.ipynb
│   ├── 02_translation_colab.ipynb
│   ├── 03_training_kaggle.ipynb
│   └── 04_gguf_conversion_colab.ipynb
│
├── app/
│   ├── server.py
│   └── swasthya_ui.html
│
├── models/
│   └── .gitkeep
│
├── requirements.txt
└── README.md
```

---

## 🔧 Configuration

**Model settings** (`app/server.py`):

| Variable | Default | Description |
|---|---|---|
| `GGUF_PATH` | `../models/swasthya-1b-q4km.gguf` | Path to GGUF model |
| `N_CTX` | `2048` | Context window size |
| `N_THREADS` | `4` | CPU threads for inference |
| `N_GPU_LAYERS` | `0` | GPU layers (set to 20+ for CUDA) |

---

## ⚠️ Known Limitations

- 1B parameter model has a quality ceiling — some queries produce inconsistent responses
- Fever-related queries sometimes generate incomplete responses due to training data patterns
- Bengali TTS requires Bengali voice pack installed on the system
- Not intended for real medical use — prototype only

---

## 🗺️ Roadmap

- [ ] Whisper-based offline STT for voice input
- [ ] Upgrade to 7B base model when compute allows
- [ ] RAG layer over verified medical guidelines (WHO/ICMR)
- [ ] Proper Android APK with WebView wrapper

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file

---

## 🙏 Acknowledgments

- **TigerLLM** - Bengali-optimized base model
- **HealthCareMagic** - Medical QA dataset
- **llama.cpp** - Inference engine
- **HuggingFace** - Model hosting and training tools
- **Kaggle** - Free T4 GPU for fine-tuning

---

## 📧 Contact

**Sourav Chandra** - souravchandra133@gmail.com

**GitHub:** [github.com/souravchandra01](https://github.com/souravchandra01)

**LinkedIn:** [linkedin.com/in/sourav-chandra-5a3112265](https://www.linkedin.com/in/sourav-chandra-5a3112265/)

---

## 🧠 Notes

This project was built as a hands-on exploration of on-device LLM deployment under real hardware constraints. The choice of a 1B model was driven by actual testing on Android — 2B quantized models exceeded available RAM on target devices. The bilingual dataset was curated manually after finding that fully automated translation introduced medical inaccuracies.

Further improvements are possible, but the current scope demonstrates a complete fine-tuning and deployment pipeline from raw data to on-device inference.

---

<div align="center">

**⭐ Star this repo if you found it helpful!**

Built with ❤️ for accessible healthcare AI

</div>