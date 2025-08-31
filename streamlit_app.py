import streamlit as st
import os
import sys
from pathlib import Path
import time
import json

# Add src directory to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

try:
    from pdf_text_extractor import PDFTextExtractor
    from text_to_audiobook import AudiobookGenerator
    from voice_cloning_interface import voice_cloning_interface
except ImportError as e:
    st.error(f"Could not import modules: {e}")
    st.stop()

def main():
    st.set_page_config(
        page_title="AI Audiobook Generator",
        page_icon="🎧",
        layout="wide"
    )
    
    # Header with status
    st.title("🎧 AI Audiobook Generator")
    
    # Status indicator
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.success("✅ System Ready - Voice Cloning Enabled")
    
    st.markdown("Transform your PDF documents into personalized audiobooks with your own voice!")
    
    # Setup directories
    data_dir = project_root / "data"
    extracted_text_dir = project_root / "extracted_text"
    output_dir = project_root / "output" / "audiobooks"
    
    # Create directories if they don't exist
    data_dir.mkdir(exist_ok=True)
    extracted_text_dir.mkdir(exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Project Structure")
        st.markdown("""
        - **data/**: Place your PDF files here
        - **extracted_text/**: Extracted and structured text
        - **output/audiobooks/**: Generated audiobook files
        - **logs/**: Generation logs
        """)
        
        st.header("🔧 Settings")
        voice_rate = st.slider("Voice Speed", 100, 300, 200, help="Words per minute")
        voice_volume = st.slider("Voice Volume", 0.1, 1.0, 0.9, help="Voice volume level")
    
    # Main content - Single unified workflow
    st.header("🚀 Complete AI Audiobook Generation Pipeline")
    st.markdown("""
    **Follow this step-by-step workflow to create your personalized audiobook:**
    
    1. **📄 Upload PDF** → Extract and structure text content
    2. **🎤 Train Voice Model** → Upload voice samples to create your voice clone
    3. **🎧 Generate Audiobook** → Create audiobook with your cloned voice
    """)
    
    # Create voice samples and models directories
    voice_samples_dir = project_root / "voice_samples"
    voice_models_dir = project_root / "voice_models"
    voice_samples_dir.mkdir(exist_ok=True)
    voice_models_dir.mkdir(exist_ok=True)
    
    # Progress tracking
    st.subheader("📊 Pipeline Progress")
    
    # Check what's available (with minimal logging)
    json_files = list(extracted_text_dir.glob("*_structured.json"))
    try:
        # Initialize generator with minimal voice model loading
        generator = AudiobookGenerator(str(output_dir))
        available_models = generator.get_available_voice_models()
    except Exception as e:
        st.error(f"Error initializing voice system: {e}")
        available_models = []
    
    # Progress indicators
    col1, col2, col3 = st.columns(3)
    with col1:
        if json_files:
            st.success("✅ Step 1: Text Extracted")
            st.write(f"📄 {len(json_files)} PDF(s) processed")
        else:
            st.warning("⏳ Step 1: Upload PDF")
    
    with col2:
        if available_models:
            st.success("✅ Step 2: Voice Model Ready")
            st.write(f"🎭 {len(available_models)} model(s) trained")
        else:
            st.warning("⏳ Step 2: Train Voice Model")
    
    with col3:
        if json_files and available_models:
            st.info("🎯 Step 3: Ready to Generate!")
        else:
            st.warning("⏳ Step 3: Complete Steps 1 & 2")
    
    st.markdown("---")
    
    # Step 1: PDF Upload and Text Extraction
    st.subheader("📄 Step 1: Upload PDF and Extract Text")
    
    uploaded_file = st.file_uploader(
        "Upload a PDF file to extract text content",
        type=['pdf'],
        help="Upload a PDF file to extract and structure text for audiobook generation"
    )
    
    if uploaded_file is not None:
        # Save uploaded file
        pdf_path = data_dir / uploaded_file.name
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"✅ PDF uploaded: {uploaded_file.name}")
        
        if st.button("🔍 Extract Text from PDF", key="extract_text"):
            with st.spinner("Extracting and structuring text from PDF..."):
                try:
                    extractor = PDFTextExtractor(str(extracted_text_dir))
                    summary = extractor.process_pdf(str(pdf_path))
                    
                    # Use the returned summary information
                    json_file = Path(summary['structured_json'])
                    
                    if json_file.exists():
                        st.success("🎉 Text extraction completed successfully!")
                        
                        # Show extracted content preview using summary
                        with st.expander("📖 Preview Extracted Content"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("📚 Title", summary.get('output_name', 'Unknown'))
                            with col2:
                                st.metric("📖 Sections", summary.get('total_sections', summary.get('total_chapters', 0)))
                            with col3:
                                st.metric("📝 Words", f"{summary.get('total_words', 0):,}")
                            
                            # Show section titles from the summary
                            sections_key = 'sections' if 'sections' in summary else 'chapters'
                            if sections_key in summary:
                                st.write("**Section Titles:**")
                                for i, section_title in enumerate(summary[sections_key], 1):
                                    st.write(f"{i}. {section_title}")
                        
                        st.info("✅ Step 1 Complete! Now proceed to Step 2 to train your voice model.")
                    else:
                        st.error("❌ Text extraction failed!")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Show available extracted texts
    if json_files:
        with st.expander("📄 Available Extracted Texts"):
            for json_file in json_files:
                st.write(f"📄 {json_file.name}")
    
    st.markdown("---")
    
    # Step 2: Voice Model Training
    st.subheader("🎤 Step 2: Train Your Voice Model")
    st.markdown("""
    Upload clear audio samples of your voice to create a personalized voice model:
    - **Recommended**: 5-15 clear voice samples
    - **Duration**: 10-60 seconds each
    - **Quality**: Clear recording, minimal background noise
    - **Content**: Varied vocabulary and emotional range
    """)
    
    # Voice sample upload
    uploaded_samples = st.file_uploader(
        "Upload voice samples (WAV/MP3/M4A/FLAC)",
        type=['wav', 'mp3', 'm4a', 'flac'],
        accept_multiple_files=True,
        help="Upload high-quality voice samples for training"
    )
    
    model_name = st.text_input(
        "Voice Model Name",
        placeholder="my_voice_model",
        help="Enter a unique name for your voice model"
    )
    
    if uploaded_samples and model_name:
        # Save uploaded samples
        sample_paths = []
        sample_dir = voice_samples_dir / model_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        for sample in uploaded_samples:
            sample_path = sample_dir / sample.name
            with open(sample_path, "wb") as f:
                f.write(sample.getbuffer())
            sample_paths.append(sample_path)
        
        st.success(f"✅ {len(uploaded_samples)} voice samples uploaded successfully!")
        
        # Show sample information
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Samples", len(uploaded_samples))
        with col2:
            total_size = sum(len(sample.getbuffer()) for sample in uploaded_samples)
            st.metric("💾 Total Size", f"{total_size / (1024*1024):.1f} MB")
        with col3:
            st.metric("📁 Model Name", model_name)
        
        # Voice analysis
        if st.button("🔍 Analyze Voice Samples", key="analyze_voice"):
            with st.spinner("Analyzing voice characteristics..."):
                try:
                    generator = AudiobookGenerator(str(output_dir))
                    
                    # Analyze first sample for preview
                    analysis = generator.analyze_voice_sample(str(sample_paths[0]))
                    
                    st.subheader("📊 Voice Analysis Results")
                    
                    # Enhanced metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🎵 Pitch", f"{analysis.get('average_pitch', 0):.1f} Hz")
                    with col2:
                        st.metric("📢 Volume", f"{analysis.get('average_volume', 0):.3f}")
                    with col3:
                        st.metric("⏱️ Duration", f"{analysis.get('duration', 0):.1f}s")
                    with col4:
                        gender = analysis.get('estimated_gender', 'unknown')
                        st.metric("👤 Gender", gender.title())
                    
                    # Quality assessment
                    quality = analysis.get('voice_quality_score', 0.7)
                    st.subheader("⭐ Quality Assessment")
                    st.progress(quality)
                    st.write(f"Voice Quality Score: {quality:.2f}/1.0")
                    
                    if quality > 0.8:
                        st.success("🎉 Excellent voice quality! Perfect for training.")
                    elif quality > 0.6:
                        st.info("👍 Good voice quality, suitable for training.")
                    else:
                        st.warning("⚠️ Lower quality detected. Consider re-recording in a quieter environment.")
                    
                except Exception as e:
                    st.error(f"❌ Analysis error: {str(e)}")
        
        # Train voice model
        if st.button("🚀 Train Voice Model", key="train_voice"):
            with st.spinner("Training voice model... This may take several minutes."):
                try:
                    generator = AudiobookGenerator(str(output_dir))
                    model_path = generator.train_voice_model(
                        str(sample_dir), 
                        model_name
                    )
                    
                    st.success(f"🎉 Voice model '{model_name}' trained successfully!")
                    st.info(f"📁 Model saved to: {model_path}")
                    st.balloons()
                    st.info("✅ Step 2 Complete! Now proceed to Step 3 to generate your audiobook.")
                    
                except Exception as e:
                    st.error(f"❌ Training error: {str(e)}")
    
    # Show available voice models
    if available_models:
        with st.expander("🎭 Your Trained Voice Models"):
            # Get detailed model information
            try:
                custom_models_info = generator.get_custom_voice_models_with_info()
                
                if custom_models_info:
                    st.write(f"**Found {len(custom_models_info)} custom voice models:**")
                    
                    for model_info in custom_models_info:
                        with st.container():
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.write(f"🎭 **{model_info['name']}**")
                                st.write(f"📊 Quality: {model_info['quality']:.2f}")
                            
                            with col2:
                                st.write(f"👤 Gender: {model_info['gender'].title()}")
                                st.write(f"📄 Samples: {model_info['samples']}")
                            
                            with col3:
                                duration = model_info['duration']
                                st.write(f"⏱️ Duration: {duration:.1f}s")
                                st.write(f"📅 Created: {model_info['created']}")
                            
                            with col4:
                                # Quality indicator
                                quality = model_info['quality']
                                if quality > 0.8:
                                    st.success("⭐ Excellent")
                                elif quality > 0.6:
                                    st.info("👍 Good")
                                else:
                                    st.warning("⚠️ Basic")
                            
                            st.write(f"📝 {model_info['description']}")
                            st.divider()
                
                else:
                    # Show basic model list
                    for i, model in enumerate(available_models, 1):
                        try:
                            model_info = generator.get_voice_model_info(model)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write(f"🎭 **{model}**")
                            with col2:
                                if model_info:
                                    quality = model_info.get('training_quality', 0)
                                    st.write(f"Quality: {quality:.2f}")
                            with col3:
                                if model_info:
                                    model_type = model_info.get('model_type', 'unknown')
                                    st.write(f"Type: {model_type}")
                        except:
                            st.write(f"🎭 **{model}** - Available")
                            
            except Exception as e:
                st.error(f"Error loading detailed model information: {e}")
                # Fallback to simple list
                for i, model in enumerate(available_models, 1):
                    st.write(f"{i}. 🎭 **{model}**")
    else:
        st.info("ℹ️ No voice models found. Train a voice model in Step 2 to see it here.")
    
    st.markdown("---")
    
    # Step 3: Generate Personalized Audiobook
    st.subheader("🎧 Step 3: Generate Your Personalized Audiobook")
    
    if not json_files:
        st.warning("⚠️ Please complete Step 1: Extract text from a PDF first.")
    elif not available_models:
        st.warning("⚠️ Please complete Step 2: Train a voice model first.")
    else:
        st.success("🎯 Ready to generate your personalized audiobook!")
        
        # Select text file
        selected_file = st.selectbox(
            "Select extracted text for audiobook:",
            json_files,
            format_func=lambda x: x.name.replace('_structured.json', ''),
            help="Choose which extracted text to convert to audiobook"
        )
        
        # Select voice model with enhanced information
        st.subheader("🎭 Select Your Voice Model")
        
        # Get detailed voice model information
        try:
            custom_models_info = generator.get_custom_voice_models_with_info()
            
            if custom_models_info:
                st.success(f"🎉 Found {len(custom_models_info)} of your trained voice models!")
                
                # Create options with detailed information
                voice_options = ["Default High-Quality Voice"]
                voice_mapping = {"Default High-Quality Voice": "Default Voice"}
                
                for model_info in custom_models_info:
                    model_name = model_info['name']
                    quality = model_info['quality']
                    gender = model_info['gender']
                    samples = model_info['samples']
                    
                    # Create descriptive option text
                    option_text = f"🎭 {model_name} (Quality: {quality:.2f}, {gender.title()}, {samples} samples)"
                    voice_options.append(option_text)
                    voice_mapping[option_text] = model_name
                
                # Additional models (non-custom)
                other_models = [m for m in available_models if not any(cm['name'] == m for cm in custom_models_info)]
                for model in other_models:
                    if model not in ['default', 'system', 'fallback']:
                        option_text = f"🎪 {model} (Pre-trained)"
                        voice_options.append(option_text)
                        voice_mapping[option_text] = model
                
                selected_voice_option = st.selectbox(
                    "Choose your voice model:",
                    voice_options,
                    help="Select your trained voice model or use the default high-quality voice"
                )
                
                selected_voice = voice_mapping[selected_voice_option]
                
                # Show selected model details
                if selected_voice != "Default Voice":
                    # Find the selected model info
                    selected_model_info = next((m for m in custom_models_info if m['name'] == selected_voice), None)
                    
                    if selected_model_info:
                        with st.expander("🔍 Selected Voice Model Details"):
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("🎭 Model", selected_model_info['name'])
                            with col2:
                                st.metric("⭐ Quality", f"{selected_model_info['quality']:.2f}")
                            with col3:
                                st.metric("👤 Gender", selected_model_info['gender'].title())
                            with col4:
                                st.metric("📄 Samples", selected_model_info['samples'])
                            
                            st.write(f"**Description:** {selected_model_info['description']}")
                            st.write(f"**Duration:** {selected_model_info['duration']:.1f} seconds")
                            st.write(f"**Created:** {selected_model_info['created']}")
                
            else:
                # Fallback to simple selection
                selected_voice = st.selectbox(
                    "Select voice model:",
                    ["Default Voice"] + available_models,
                    help="Choose your trained voice model or use default"
                )
                
                if selected_voice != "Default Voice":
                    st.info(f"🎭 Using voice model: **{selected_voice}**")
                    
        except Exception as e:
            st.error(f"Error loading voice model details: {e}")
            # Simple fallback
            selected_voice = st.selectbox(
                "Select voice model:",
                ["Default Voice"] + available_models,
                help="Choose your trained voice model or use default"
            )
        
        # Show selection summary
        with st.expander("📋 Generation Summary"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Text Source:** {selected_file.name.replace('_structured.json', '')}")
                # Load JSON to show details
                try:
                    with open(selected_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    total_sections = len(data.get('sections', data.get('chapters', [])))
                    st.write(f"**Sections:** {total_sections}")
                except:
                    pass
            with col2:
                st.write(f"**Voice Model:** {selected_voice}")
                if selected_voice != "Default Voice":
                    try:
                        # Get model info for display
                        model_info = generator.get_voice_model_info(selected_voice)
                        if model_info:
                            quality = model_info.get('training_quality', 0)
                            gender = model_info.get('characteristics', {}).get('estimated_gender', 'unknown')
                            st.write(f"**Type:** Personalized ({gender.title()}, Quality: {quality:.2f})")
                        else:
                            st.write("**Type:** Personalized Cloned Voice")
                    except:
                        st.write("**Type:** Personalized Cloned Voice")
                else:
                    st.write("**Type:** High-Quality Default Voice")
        
        # Generate audiobook
        if st.button("🎬 Generate Personalized Audiobook", key="generate_final"):
            with st.spinner("Generating your personalized audiobook... This may take several minutes."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    generator = AudiobookGenerator(str(output_dir))
                    
                    if selected_voice == "Default Voice":
                        # Use regular audiobook generation
                        status_text.text("🎙️ Generating with high-quality default voice...")
                        progress_bar.progress(0.5)
                        report = generator.create_audiobook_from_json(str(selected_file))
                        voice_type = "Default High-Quality Voice"
                    else:
                        # Use cloned voice
                        status_text.text(f"🎭 Generating with your cloned voice: {selected_voice}...")
                        progress_bar.progress(0.3)
                        audiobook_path = generator.create_cloned_audiobook_from_json(
                            str(selected_file), 
                            voice_model_name=selected_voice
                        )
                        
                        progress_bar.progress(0.8)
                        
                        # Create report for cloned voice audiobook
                        audiobook_file = Path(audiobook_path)
                        file_size = audiobook_file.stat().st_size / (1024*1024)
                        
                        report = {
                            'title': audiobook_file.stem,
                            'audiobook_file': audiobook_file.name,
                            'audiobook_path': str(audiobook_path),
                            'file_size_mb': f"{file_size:.1f}",
                            'voice_model': selected_voice,
                            'personalized': True
                        }
                        voice_type = f"Personalized Cloned Voice ({selected_voice})"
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Audiobook generation completed!")
                    
                    if report:
                        st.success("🎉 Your personalized audiobook has been generated successfully!")
                        
                        # Show comprehensive results
                        st.subheader("📊 Audiobook Generation Results")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("📚 Title", report['title'])
                        with col2:
                            st.metric("💾 File Size", f"{report['file_size_mb']} MB")
                        with col3:
                            st.metric("🎭 Voice Type", voice_type)
                        with col4:
                            if report.get('personalized'):
                                st.metric("⭐ Quality", "Personalized")
                            else:
                                st.metric("⭐ Quality", "High-Quality")
                        
                        # Additional metrics if available
                        if 'total_duration_minutes' in report:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("⏱️ Duration", f"{report['total_duration_minutes']:.1f} min")
                            with col2:
                                st.metric("📖 Sections", report.get('total_sections', report.get('total_chapters', 0)))
                            with col3:
                                st.metric("📝 Words", f"{report.get('total_words', 0):,}")
                        
                        # Download section
                        st.subheader("📥 Download Your Audiobook")
                        audiobook_path = Path(report['audiobook_path'])
                        
                        if audiobook_path.exists():
                            # Success message
                            if report.get('personalized'):
                                st.success("🚀 Your personalized audiobook with cloned voice is ready!")
                                st.balloons()
                            else:
                                st.success("🎉 Your high-quality audiobook is ready!")
                            
                            # Download button with styling
                            with open(audiobook_path, 'rb') as f:
                                st.download_button(
                                    label="🎧 Download Your Personalized Audiobook",
                                    data=f.read(),
                                    file_name=report['audiobook_file'],
                                    mime="audio/wav",
                                    key="final_audiobook_download",
                                    help="Click to download your completed audiobook"
                                )
                            
                            # Show file information
                            st.info(f"📁 Audiobook saved as: `{report['audiobook_file']}`")
                            
                        else:
                            st.error("❌ Audiobook file not found after generation!")
                    else:
                        st.error("❌ Audiobook generation failed!")
                        
                except Exception as e:
                    st.error(f"❌ Generation error: {str(e)}")
                    progress_bar.progress(0)
                    status_text.text("❌ Generation failed")
    
    st.markdown("---")
    
    # Quick Actions Section
    st.subheader("⚡ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Clear All Data"):
            if st.confirm("Are you sure you want to clear all extracted texts and voice models?"):
                # Clear extracted texts
                for file in extracted_text_dir.glob("*"):
                    if file.is_file():
                        file.unlink()
                
                # Clear voice models
                for file in voice_models_dir.glob("*"):
                    if file.is_file():
                        file.unlink()
                
                # Clear voice samples
                for folder in voice_samples_dir.glob("*"):
                    if folder.is_dir():
                        for file in folder.glob("*"):
                            file.unlink()
                        folder.rmdir()
                
                st.success("🧹 All data cleared!")
                st.experimental_rerun()
    
    with col2:
        if json_files:
            st.info(f"📄 {len(json_files)} Text(s) Available")
        else:
            st.warning("📄 No Texts Available")
    
    with col3:
        if available_models:
            st.info(f"🎭 {len(available_models)} Voice Model(s)")
        else:
            st.warning("🎭 No Voice Models")
    
    # Footer
    st.markdown("---")
    st.markdown("**AI Audiobook Generator** - Convert PDFs to audiobooks with ease!")

if __name__ == "__main__":
    main()
