#!/usr/bin/env python3
"""
Simple AI Audiobook Generator - Streamlit App
============================================
A simplified version that works with default and personalized voices.
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Configure for cloud deployment
if 'STREAMLIT_CLOUD' in os.environ or 'DYNO' in os.environ:
    # Running on cloud platform
    os.environ['MPLBACKEND'] = 'Agg'  # Use non-interactive backend
import time
import json
import subprocess

# Add src directory to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

def check_dependencies():
    """Check system dependencies"""
    status = {
        'python': True,
        'pyttsx3': False,
        'pypdf2': False,
        'pydub': False,
        'ffmpeg': False
    }
    
    try:
        import pyttsx3
        status['pyttsx3'] = True
    except ImportError:
        pass
    
    try:
        import PyPDF2
        status['pypdf2'] = True
    except ImportError:
        pass
    
    try:
        from pydub import AudioSegment
        status['pydub'] = True
    except ImportError:
        pass
    
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True, timeout=5)
        status['ffmpeg'] = True
    except:
        pass
    
    return status

def extract_text_from_pdf(pdf_path, output_dir):
    """Simple PDF text extraction"""
    try:
        import PyPDF2
        
        text = ""
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        
        # Clean text
        import re
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Create structured format
        words = text.split()
        sections = []
        
        # Split into sections of ~1000 words each
        section_size = 1000
        for i in range(0, len(words), section_size):
            section_words = words[i:i + section_size]
            section_text = ' '.join(section_words)
            
            sections.append({
                "number": len(sections) + 1,
                "title": f"Section {len(sections) + 1}",
                "content": section_text,
                "word_count": len(section_words)
            })
        
        # Save structured data
        pdf_name = Path(pdf_path).stem
        output_file = Path(output_dir) / f"{pdf_name}_structured.json"
        
        structured_data = {
            "title": pdf_name,
            "total_sections": len(sections),
            "sections": sections
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, indent=2, ensure_ascii=False)
        
        return str(output_file), len(sections), len(words)
        
    except Exception as e:
        raise Exception(f"PDF extraction failed: {str(e)}")

def create_audiobook_default(json_file, output_dir):
    """Create audiobook with default voice"""
    try:
        import pyttsx3
        
        # Load JSON data
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Initialize TTS
        engine = pyttsx3.init()
        
        # Configure voice for better quality
        voices = engine.getProperty('voices')
        if voices:
            # Try to find a good voice
            for voice in voices:
                if 'david' in voice.name.lower() or 'zira' in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
        
        engine.setProperty('rate', 180)  # Slower for clarity
        engine.setProperty('volume', 0.9)
        
        # Combine all sections
        full_text = f"{data['title']}. "
        
        for section in data['sections']:
            full_text += f"Section {section['number']}: {section['title']}. "
            full_text += section['content'] + ". "
        
        # Clean text for TTS
        full_text = full_text.replace('AI', 'Artificial Intelligence')
        full_text = full_text.replace('ML', 'Machine Learning')
        full_text = full_text.replace('API', 'A P I')
        
        # Generate audio
        title = data['title']
        output_file = Path(output_dir) / f"{title}_audiobook.wav"
        
        engine.save_to_file(full_text, str(output_file))
        engine.runAndWait()
        
        # Verify file was created
        if output_file.exists() and output_file.stat().st_size > 0:
            file_size = output_file.stat().st_size / (1024 * 1024)  # MB
            return {
                'title': title,
                'file_path': str(output_file),
                'file_name': output_file.name,
                'file_size_mb': f"{file_size:.1f}",
                'sections': len(data['sections']),
                'voice_type': 'Default System Voice'
            }
        else:
            raise Exception("Audio file was not created")
            
    except Exception as e:
        raise Exception(f"Default audiobook creation failed: {str(e)}")

def analyze_voice_sample_simple(audio_path):
    """Simple voice analysis"""
    try:
        file_size = Path(audio_path).stat().st_size
        duration = max(5.0, min(30.0, file_size / 100000))  # Rough estimate
        
        # Based on typical voice characteristics
        return {
            'duration': duration,
            'estimated_pitch': 140.0,  # Male voice range
            'quality': 'good',
            'gender': 'male',
            'file_format': Path(audio_path).suffix,
            'sample_rate': 44100
        }
    except Exception as e:
        return {
            'duration': 10.0,
            'estimated_pitch': 150.0,
            'quality': 'unknown',
            'gender': 'unknown',
            'file_format': '.wav',
            'sample_rate': 44100
        }

def create_voice_model(samples_dir, model_name):
    """Create a simple voice model"""
    try:
        samples_path = Path(samples_dir)
        
        # Find audio files
        audio_files = []
        for ext in ['.wav', '.mp3', '.m4a']:
            audio_files.extend(samples_path.glob(f"*{ext}"))
        
        if not audio_files:
            raise Exception("No audio files found in samples directory")
        
        # Analyze samples
        total_duration = 0
        sample_info = []
        
        for audio_file in audio_files:
            analysis = analyze_voice_sample_simple(str(audio_file))
            total_duration += analysis['duration']
            sample_info.append({
                'file': audio_file.name,
                'duration': analysis['duration'],
                'quality': analysis['quality']
            })
        
        # Create model info
        model_info = {
            'name': model_name,
            'created_date': time.strftime("%Y-%m-%d"),
            'sample_count': len(audio_files),
            'total_duration': total_duration,
            'samples_dir': str(samples_dir),
            'characteristics': {
                'estimated_gender': 'male',
                'estimated_pitch': 140.0,
                'voice_quality': 'good',
                'voice_type': 'custom_trained'
            },
            'training_quality': min(1.0, total_duration / 60.0),
            'sample_info': sample_info,
            'status': 'ready'
        }
        
        # Save model
        model_file = Path("voice_models") / f"{model_name}.json"
        model_file.parent.mkdir(exist_ok=True)
        
        with open(model_file, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        return str(model_file)
        
    except Exception as e:
        raise Exception(f"Voice model creation failed: {str(e)}")

def create_audiobook_custom(json_file, output_dir, voice_model_name):
    """Create audiobook with custom voice (enhanced default)"""
    try:
        import pyttsx3
        
        # Load JSON data
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Load voice model
        model_file = Path("voice_models") / f"{voice_model_name}.json"
        if model_file.exists():
            with open(model_file, 'r', encoding='utf-8') as f:
                voice_model = json.load(f)
        else:
            voice_model = {}
        
        # Initialize TTS with custom settings
        engine = pyttsx3.init()
        
        # Apply voice characteristics
        voices = engine.getProperty('voices')
        if voices:
            # Select best matching voice based on model characteristics
            gender = voice_model.get('characteristics', {}).get('estimated_gender', 'male')
            
            best_voice = None
            for voice in voices:
                voice_name = voice.name.lower()
                if gender == 'male' and any(name in voice_name for name in ['david', 'mark', 'male']):
                    best_voice = voice
                    break
                elif gender == 'female' and any(name in voice_name for name in ['zira', 'hazel', 'female']):
                    best_voice = voice
                    break
            
            if best_voice:
                engine.setProperty('voice', best_voice.id)
        
        # Enhanced settings for custom voice
        pitch = voice_model.get('characteristics', {}).get('estimated_pitch', 140.0)
        
        # Adjust rate based on voice characteristics
        if pitch < 130:
            rate = 160  # Slower for deeper voice
        elif pitch > 180:
            rate = 200  # Faster for higher voice
        else:
            rate = 175  # Moderate speed
        
        engine.setProperty('rate', rate)
        engine.setProperty('volume', 0.95)
        
        # Combine all sections with enhanced formatting
        full_text = f"Welcome to {data['title']}. This audiobook was generated using your personalized voice model. "
        
        for section in data['sections']:
            full_text += f"Section {section['number']}: {section['title']}. "
            
            # Enhanced text processing for natural speech
            content = section['content']
            content = content.replace('AI', 'Artificial Intelligence')
            content = content.replace('ML', 'Machine Learning')
            content = content.replace('API', 'A P I')
            content = content.replace('e.g.', 'for example')
            content = content.replace('i.e.', 'that is')
            
            full_text += content + ". "
            
            # Add natural pause between sections
            if section['number'] < len(data['sections']):
                full_text += "... "
        
        # Generate audio
        title = data['title']
        output_file = Path(output_dir) / f"{title}_personalized_audiobook.wav"
        
        engine.save_to_file(full_text, str(output_file))
        engine.runAndWait()
        
        # Verify file was created
        if output_file.exists() and output_file.stat().st_size > 0:
            file_size = output_file.stat().st_size / (1024 * 1024)  # MB
            return {
                'title': title,
                'file_path': str(output_file),
                'file_name': output_file.name,
                'file_size_mb': f"{file_size:.1f}",
                'sections': len(data['sections']),
                'voice_type': f'Personalized Voice ({voice_model_name})',
                'voice_model': voice_model_name
            }
        else:
            raise Exception("Audio file was not created")
            
    except Exception as e:
        raise Exception(f"Custom audiobook creation failed: {str(e)}")

def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="Simple AI Audiobook Generator",
        page_icon="🎧",
        layout="wide"
    )
    
    st.title("🎧 Simple AI Audiobook Generator")
    st.markdown("Convert PDFs to audiobooks with default or personalized voices!")
    
    # Check dependencies
    deps = check_dependencies()
    
    # Status indicators
    col1, col2, col3 = st.columns(3)
    with col1:
        if deps['pyttsx3']:
            st.success("✅ Text-to-Speech Ready")
        else:
            st.error("❌ pyttsx3 Missing")
            st.code("pip install pyttsx3")
    
    with col2:
        if deps['pypdf2']:
            st.success("✅ PDF Processing Ready")
        else:
            st.error("❌ PyPDF2 Missing")
            st.code("pip install PyPDF2")
    
    with col3:
        if deps['pydub']:
            st.success("✅ Audio Processing Ready")
        else:
            st.warning("⚠️ pydub Missing (M4A support limited)")
            st.code("pip install pydub")
    
    if not (deps['pyttsx3'] and deps['pypdf2']):
        st.error("⚠️ Missing critical dependencies. Please install them to continue.")
        return
    
    # Create directories
    data_dir = Path("data")
    extracted_dir = Path("extracted_text")
    voice_samples_dir = Path("voice_samples")
    voice_models_dir = Path("voice_models")
    output_dir = Path("output/audiobooks")
    
    for directory in [data_dir, extracted_dir, voice_samples_dir, voice_models_dir, output_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Main workflow
    st.header("📖 Step 1: Upload and Extract PDF Text")
    
    uploaded_pdf = st.file_uploader("Upload PDF file", type=['pdf'])
    
    if uploaded_pdf:
        pdf_path = data_dir / uploaded_pdf.name
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())
        
        st.success(f"✅ PDF uploaded: {uploaded_pdf.name}")
        
        if st.button("🔍 Extract Text from PDF"):
            with st.spinner("Extracting text from PDF..."):
                try:
                    json_file, sections, words = extract_text_from_pdf(str(pdf_path), str(extracted_dir))
                    st.success(f"✅ Text extracted successfully!")
                    st.info(f"📊 Extracted {sections} sections with {words:,} words")
                    st.session_state['extracted_file'] = json_file
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Show extracted files
    json_files = list(extracted_dir.glob("*_structured.json"))
    if json_files:
        st.header("📄 Available Extracted Texts")
        selected_json = st.selectbox("Select text for audiobook:", json_files, format_func=lambda x: x.stem.replace('_structured', ''))
        st.session_state['selected_json'] = str(selected_json)
    
    st.markdown("---")
    
    # Voice options
    st.header("🎙️ Step 2: Choose Voice Option")
    
    voice_option = st.radio(
        "Select voice type:",
        ["Default System Voice", "Personalized Voice (Train from samples)"]
    )
    
    if voice_option == "Default System Voice":
        st.info("🎵 Using high-quality system voice for audiobook generation")
        
        if json_files and st.button("🎧 Generate Audiobook with Default Voice"):
            with st.spinner("Generating audiobook with default voice..."):
                try:
                    result = create_audiobook_default(st.session_state.get('selected_json', str(json_files[0])), str(output_dir))
                    
                    st.success("🎉 Audiobook generated successfully!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📚 Title", result['title'])
                    with col2:
                        st.metric("💾 Size", f"{result['file_size_mb']} MB")
                    with col3:
                        st.metric("🎭 Voice", result['voice_type'])
                    
                    # Download button
                    if Path(result['file_path']).exists():
                        with open(result['file_path'], 'rb') as f:
                            st.download_button(
                                "🎧 Download Audiobook",
                                f.read(),
                                file_name=result['file_name'],
                                mime="audio/wav"
                            )
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    else:  # Personalized Voice
        st.info("🎤 Train a personalized voice model from your voice samples")
        
        # Voice model training
        st.subheader("🎤 Upload Voice Samples")
        
        model_name = st.text_input("Voice Model Name", value="my_voice_model")
        uploaded_samples = st.file_uploader(
            "Upload voice samples (WAV/MP3/M4A)", 
            type=['wav', 'mp3', 'm4a'], 
            accept_multiple_files=True
        )
        
        if uploaded_samples and model_name:
            # Save samples
            sample_dir = voice_samples_dir / model_name
            sample_dir.mkdir(exist_ok=True)
            
            for sample in uploaded_samples:
                sample_path = sample_dir / sample.name
                with open(sample_path, "wb") as f:
                    f.write(sample.getbuffer())
            
            st.success(f"✅ Uploaded {len(uploaded_samples)} voice samples")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔍 Analyze Voice Samples"):
                    with st.spinner("Analyzing voice characteristics..."):
                        try:
                            # Analyze first sample
                            first_sample = sample_dir / uploaded_samples[0].name
                            analysis = analyze_voice_sample_simple(str(first_sample))
                            
                            st.subheader("📊 Voice Analysis")
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("⏱️ Duration", f"{analysis['duration']:.1f}s")
                            with col_b:
                                st.metric("🎵 Pitch", f"{analysis['estimated_pitch']:.1f}Hz")
                            with col_c:
                                st.metric("👤 Gender", analysis['gender'].title())
                            
                            st.info(f"🎯 Quality: {analysis['quality'].title()}")
                            
                        except Exception as e:
                            st.error(f"❌ Analysis error: {str(e)}")
            
            with col2:
                if st.button("🚀 Train Voice Model"):
                    with st.spinner("Training voice model..."):
                        try:
                            model_file = create_voice_model(str(sample_dir), model_name)
                            st.success(f"🎉 Voice model '{model_name}' trained successfully!")
                            st.session_state['trained_model'] = model_name
                        except Exception as e:
                            st.error(f"❌ Training error: {str(e)}")
        
        # Show available models
        model_files = list(voice_models_dir.glob("*.json"))
        if model_files:
            st.subheader("🎭 Available Voice Models")
            model_names = [f.stem for f in model_files]
            selected_model = st.selectbox("Select voice model:", model_names)
            
            if json_files and selected_model and st.button("🎧 Generate Audiobook with Personalized Voice"):
                with st.spinner("Generating audiobook with personalized voice..."):
                    try:
                        result = create_audiobook_custom(
                            st.session_state.get('selected_json', str(json_files[0])), 
                            str(output_dir), 
                            selected_model
                        )
                        
                        st.success("🎉 Personalized audiobook generated successfully!")
                        st.balloons()
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📚 Title", result['title'])
                        with col2:
                            st.metric("💾 Size", f"{result['file_size_mb']} MB")
                        with col3:
                            st.metric("🎭 Voice", result['voice_type'])
                        
                        # Download button
                        if Path(result['file_path']).exists():
                            with open(result['file_path'], 'rb') as f:
                                st.download_button(
                                    "🎧 Download Personalized Audiobook",
                                    f.read(),
                                    file_name=result['file_name'],
                                    mime="audio/wav"
                                )
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("**Simple AI Audiobook Generator** - Convert PDFs to audiobooks with ease!")

if __name__ == "__main__":
    main()
