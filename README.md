# 🎧 AI Audiobook Generator with Advanced Voice Cloning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)

**Transform PDF documents into personalized audiobooks with AI-powered voice cloning technology**

[Features](#-features) • [Quick Start](#-quick-start) • [Installation](#-installation) • [Documentation](#-documentation) • [Contributing](#-contributing)

</div>

---

## 🌟 Overview

The **AI Audiobook Generator** is a comprehensive solution that converts PDF documents into high-quality audiobooks using advanced voice cloning technology. Train custom voice models with your own voice samples and generate audiobooks that sound exactly like you!

### 🎯 Key Highlights

- **🎭 Advanced Voice Cloning**: Train custom voice models from 5-15 second audio samples
- **📚 Complete PDF Pipeline**: Extract, structure, and convert PDFs to audiobooks
- **🖥️ Professional Web Interface**: Intuitive Streamlit-based UI with progress tracking
- **🔊 High-Quality Audio**: Professional-grade audio processing and synthesis
- **📊 Voice Analytics**: Comprehensive voice characteristic analysis and quality assessment
- **🎛️ Model Management**: Save, reuse, and manage multiple voice models

---

## ✨ Features

### 📄 **PDF Processing**

- **Smart Text Extraction**: Advanced PDF parsing with structure preservation
- **Intelligent Chapter Detection**: Automatic section identification and organization
- **Format Support**: Supports text-based PDFs, academic papers, and e-books
- **Metadata Extraction**: Preserves document structure and formatting cues

### 🎭 **Voice Cloning Technology**

- **Custom Model Training**: Train personalized voice models from audio samples
- **Multi-Format Support**: WAV, MP3, M4A, FLAC audio input support
- **Voice Analysis**: Pitch detection, gender estimation, quality assessment
- **Model Persistence**: Save and reuse trained voice models across sessions

### 🎙️ **Audio Generation**

- **Professional TTS**: High-quality text-to-speech with multiple engine support
- **Voice Synthesis**: Generate speech using your cloned voice models
- **Audio Processing**: Noise reduction, normalization, and enhancement
- **Format Options**: Output in WAV format with customizable parameters

### 🖥️ **User Experience**

- **Unified Workflow**: Step-by-step guided process from PDF to audiobook
- **Real-time Feedback**: Progress tracking and detailed status updates
- **Model Showcase**: Enhanced dropdown with voice model details and quality scores
- **Error Handling**: Robust error management with helpful user guidance

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** (3.11 recommended)
- **Windows 10/11** (Linux/macOS supported with minor configuration)
- **4GB RAM minimum** (8GB recommended for voice cloning)
- **2GB free disk space**

### 🔧 Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/UsmanAamir01/Audiobook-Voice-Cloner.git
   cd Audiobook-Voice-Cloner
   ```

2. **Create Virtual Environment**

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Linux/macOS
   source venv/bin/activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Application**

   ```bash
   streamlit run streamlit_app.py
   ```

5. **Open in Browser**
   - Navigate to `http://localhost:8501`
   - Start creating your audiobooks!

---

## 📱 Using the Application

### Step 1: PDF Upload & Text Extraction

1. Upload your PDF document
2. Click "Extract Text from PDF"
3. Review extracted content and sections

### Step 2: Voice Model Training

1. Record 5-15 voice samples (10-60 seconds each)
2. Upload audio files (WAV, MP3, M4A supported)
3. Train your personalized voice model
4. Review voice analysis and quality metrics

### Step 3: Audiobook Generation

1. Select your trained voice model from enhanced dropdown
2. Review generation summary with model details
3. Generate your personalized audiobook
4. Download the completed audiobook file

---

## 🏗️ Architecture

### 📁 Project Structure

```
Audiobook-Voice-Cloner/
├── 📱 streamlit_app.py              # Main Streamlit application
├── 📋 requirements.txt              # Python dependencies
├── 📖 README.md                     # Project documentation
├── 🔒 .gitignore                    # Git ignore rules
│
├── 📁 src/                          # Core source code
│   ├── 🔍 pdf_text_extractor.py         # PDF processing engine
│   ├── 🎙️ text_to_audiobook.py          # Audio generation system
│   ├── 🎭 voice_cloner.py                # Voice cloning engine
│   └── 🖥️ voice_cloning_interface.py     # UI components
│
├── 📁 data/                         # Input PDF files
├── 📁 extracted_text/               # Processed text files
├── 📁 voice_models/                 # Trained voice models
├── 📁 voice_samples/                # Training audio samples
├── 📁 output/audiobooks/            # Generated audiobooks
├── 📁 logs/                         # Application logs
└── 📁 venv/                         # Virtual environment
```

### 🔧 Core Components

#### **PDF Text Extractor** (`pdf_text_extractor.py`)

- Advanced PDF parsing with PyMuPDF
- Intelligent section detection algorithms
- Structured data output (JSON format)
- Comprehensive error handling and logging

#### **Voice Cloner** (`voice_cloner.py`)

- Multi-engine TTS support (pyttsx3, Tortoise TTS)
- Audio preprocessing and analysis
- Voice characteristic extraction
- Model training and persistence

#### **Audiobook Generator** (`text_to_audiobook.py`)

- Text-to-speech synthesis
- Voice model integration
- Audio post-processing
- Quality optimization

#### **Streamlit Interface** (`streamlit_app.py`)

- Unified workflow implementation
- Enhanced UI components
- Real-time progress tracking
- Professional voice model management

---

## ⚙️ Configuration

### Audio Settings

```python
# Voice Parameters
VOICE_RATE = 170        # Words per minute (100-300)
VOICE_VOLUME = 0.9      # Volume level (0.1-1.0)
SAMPLE_RATE = 22050     # Audio sample rate
AUDIO_FORMAT = "WAV"    # Output format

# Voice Cloning
MIN_SAMPLES = 5         # Minimum training samples
MAX_DURATION = 60       # Maximum sample duration (seconds)
QUALITY_THRESHOLD = 0.6 # Minimum quality score
```

### Model Training

```python
# Training Parameters
CHUNK_SIZE = 100        # Words per processing chunk
TIMEOUT_SECONDS = 15    # TTS generation timeout
ESTIMATION_MODE = True  # Use estimation for unsupported formats
```

---

## 📊 Voice Model Management

### Model Information

Each trained voice model includes:

- **Quality Score**: 0.0-1.0 based on duration and clarity
- **Voice Characteristics**: Pitch analysis, gender estimation
- **Sample Count**: Number of training samples used
- **Training Duration**: Total audio duration processed
- **Creation Date**: Model training timestamp

### Enhanced Dropdown Display

```
🎭 my_voice (Quality: 0.85, Male, 12 samples)
🎭 professional_voice (Quality: 0.92, Female, 8 samples)
Default High-Quality Voice
```

---

## 🔬 Technical Specifications

### Supported Formats

#### Input

- **PDF**: Text-based documents (not scanned images)
- **Audio**: WAV, MP3, M4A, FLAC for voice training

#### Output

- **Audiobook**: High-quality WAV files
- **Metadata**: JSON reports with generation details
- **Models**: Persistent voice model files

### Performance Metrics

| Feature          | Processing Time    | Quality        |
| ---------------- | ------------------ | -------------- |
| PDF Extraction   | 2-10 seconds       | High           |
| Voice Training   | 30-120 seconds     | Professional   |
| Audio Generation | 0.1-0.5x real-time | Studio Quality |

---

## 🐛 Troubleshooting

### Common Issues

#### **No Voice Models Detected**

```bash
# Check voice models directory
ls voice_models/

# Verify model files
python -c "from src.voice_cloner import VoiceCloner; vc = VoiceCloner(); print(vc.get_available_models())"
```

#### **M4A File Processing Warnings**

- These are normal - the system uses estimation for M4A files
- For best results, use WAV or MP3 format for voice samples
- Quality is maintained through intelligent fallback processing

#### **TTS Engine Issues**

```bash
# Test TTS system
python -c "import pyttsx3; engine = pyttsx3.init(); engine.say('Test'); engine.runAndWait()"
```

### Performance Optimization

1. **Use SSD storage** for faster file operations
2. **Close audio applications** during generation
3. **Monitor memory usage** for large documents
4. **Use high-quality voice samples** for better model training

---

## 🔮 Advanced Features

### Voice Cloning Engines

#### **Fallback Engine (Default)**

- Reliable cross-platform TTS
- Fast processing and generation
- No additional dependencies required

#### **Tortoise TTS (Optional)**

- State-of-the-art voice cloning
- Requires additional setup and resources
- Superior voice quality and naturalness

### API Integration

```python
from src.voice_cloner import VoiceCloner
from src.text_to_audiobook import AudiobookGenerator

# Initialize components
cloner = VoiceCloner()
generator = AudiobookGenerator()

# Train voice model
model_path = cloner.train_voice_model("samples/", "my_voice")

# Generate audiobook
audiobook = generator.create_cloned_audiobook_from_json(
    "text.json",
    voice_model_name="my_voice"
)
```

---

## 📈 Roadmap

### Version 2.0 (Planned)

- [ ] **Multi-language Support**: International TTS engines
- [ ] **Batch Processing**: Multiple PDF handling
- [ ] **Cloud Integration**: Remote model training
- [ ] **Voice Marketplace**: Share and download voice models
- [ ] **Mobile App**: iOS/Android companion app

### Version 2.1 (Future)

- [ ] **Real-time Preview**: Live audio generation preview
- [ ] **Advanced Editing**: Chapter-level voice customization
- [ ] **Background Music**: Soundtrack integration
- [ ] **Team Collaboration**: Multi-user voice model sharing

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/UsmanAamir01/Audiobook-Voice-Cloner.git

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black src/
flake8 src/
```

### Contribution Areas

- **🐛 Bug Fixes**: Improve stability and reliability
- **✨ New Features**: Enhance functionality and user experience
- **📚 Documentation**: Improve guides and API documentation
- **🔧 Performance**: Optimize processing and memory usage
- **🎨 UI/UX**: Enhance the Streamlit interface

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses

- **PyMuPDF**: AGPL-3.0 License
- **pyttsx3**: MPL-2.0 License
- **Streamlit**: Apache-2.0 License

---

## 🙏 Acknowledgments

- **Streamlit Team** for the excellent web framework
- **PyMuPDF Contributors** for robust PDF processing
- **pyttsx3 Maintainers** for reliable TTS functionality
- **Open Source Community** for continuous inspiration and support

---

## 📞 Support & Community

### Getting Help

- 📖 **Documentation**: Comprehensive guides and API reference
- 🐛 **Issues**: [GitHub Issues](https://github.com/UsmanAamir01/Audiobook-Voice-Cloner/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/UsmanAamir01/Audiobook-Voice-Cloner/discussions)
- 📧 **Email**: support@ai-audiobook-generator.com

### Community

- 🌟 **Star the Project** if you find it useful
- 🍴 **Fork and Customize** for your specific needs
- 📢 **Share Your Experience** with the community

---

<div align="center">

**🎧 Happy Audiobook Generation! 📚**

Made with ❤️ by the AI Audiobook Generator Team

[⬆ Back to Top](#-ai-audiobook-generator-with-advanced-voice-cloning)

</div>
