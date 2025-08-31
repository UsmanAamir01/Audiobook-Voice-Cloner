#!/usr/bin/env python3
"""
Simple Text to Audiobook Converter
=================================
Converts structured text (JSON) to audiobook with WAV files for each section.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List
import logging

# Basic TTS
try:
    import pyttsx3
except ImportError as e:
    raise ImportError("pyttsx3 is required. Please install it with: pip install pyttsx3") from e

# Audio processing (optional)
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

# Voice cloning integration
try:
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from voice_cloner import VoiceCloner
    VOICE_CLONING_AVAILABLE = True
except ImportError:
    VOICE_CLONING_AVAILABLE = False

class AudiobookGenerator:
    """Simple and reliable audiobook generator using pyttsx3 with voice cloning support"""
    
    def __init__(self, output_dir: str = "output/audiobooks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging with proper path handling
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'audiobook_generation.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Setup TTS
        self.setup_tts()
        
        # Initialize voice cloner if available
        if VOICE_CLONING_AVAILABLE:
            try:
                # Use correct paths for voice cloner
                voice_models_dir = Path("voice_models")
                voice_samples_dir = Path("voice_samples")
                self.voice_cloner = VoiceCloner(str(voice_models_dir), str(voice_samples_dir))
                self.logger.info("Voice cloning capabilities enabled")
            except Exception as e:
                self.logger.warning(f"Voice cloning initialization failed: {e}")
                self.voice_cloner = None
        else:
            self.voice_cloner = None
    
    def setup_tts(self):
        """Setup pyttsx3 TTS engine"""
        self.logger.info("Setting up TTS engine...")
        
        try:
            self.tts_engine = pyttsx3.init()
            
            # Get available voices
            voices = self.tts_engine.getProperty('voices')
            
            if voices:
                self.logger.info(f"Available voices: {len(voices)}")
                
                # Try to find a good voice
                best_voice = None
                for voice in voices:
                    voice_name = voice.name.lower()
                    # Prefer neural or high-quality voices
                    if any(keyword in voice_name for keyword in ['neural', 'enhanced', 'premium', 'david', 'zira']):
                        best_voice = voice
                        break
                
                if best_voice:
                    self.tts_engine.setProperty('voice', best_voice.id)
                    self.logger.info(f"Selected voice: {best_voice.name}")
                else:
                    # Use first available voice
                    self.tts_engine.setProperty('voice', voices[0].id)
                    self.logger.info(f"Using default voice: {voices[0].name}")
            
            # Set speech parameters for better quality
            self.tts_engine.setProperty('rate', 170)    # Speed (words per minute)
            self.tts_engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
            
            self.logger.info("TTS engine configured successfully")
            
        except Exception as e:
            self.logger.error(f"TTS setup failed: {e}")
            raise Exception("Could not initialize TTS engine")
    
    def clean_text_for_tts(self, text: str) -> str:
        """Clean and prepare text for TTS"""
        # Common abbreviations and their pronunciations
        replacements = {
            'AI': 'Artificial Intelligence',
            'ML': 'Machine Learning',
            'DL': 'Deep Learning',
            'NLP': 'Natural Language Processing',
            'GPU': 'Graphics Processing Unit',
            'CPU': 'Central Processing Unit',
            'API': 'Application Programming Interface',
            'HTTP': 'H T T P',
            'URL': 'U R L',
            'PDF': 'P D F',
            'e.g.': 'for example',
            'i.e.': 'that is',
            'etc.': 'and so on',
            'vs.': 'versus',
            'Mr.': 'Mister',
            'Dr.': 'Doctor',
            'Prof.': 'Professor'
        }
        
        # Apply replacements
        for abbrev, full in replacements.items():
            text = text.replace(abbrev, full)
        
        # Add natural pauses
        text = text.replace('.', '. ')
        text = text.replace(',', ', ')
        text = text.replace(';', '; ')
        text = text.replace(':', ': ')
        
        # Clean up extra spaces
        import re
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def generate_audio_for_text(self, text: str, output_path: str) -> bool:
        """Generate audio file from text"""
        try:
            # Clean text for better TTS
            clean_text = self.clean_text_for_tts(text)
            
            # Generate audio
            self.tts_engine.save_to_file(clean_text, output_path)
            self.tts_engine.runAndWait()
            
            # Verify file was created
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                self.logger.info(f"Generated: {output_path}")
                return True
            else:
                self.logger.error(f"Audio file not created or empty: {output_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error generating audio: {e}")
            return False
    
    def get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in minutes"""
        try:
            if PYDUB_AVAILABLE:
                audio = AudioSegment.from_wav(audio_path)
                return len(audio) / 1000.0 / 60.0  # Convert to minutes
            else:
                # Fallback: estimate based on file size (very rough)
                file_size = os.path.getsize(audio_path)
                # Rough estimate: 1 MB ≈ 1 minute for WAV
                return file_size / (1024 * 1024)
        except:
            return 0.0
    
    def create_audiobook_from_json(self, json_path: str) -> Dict:
        """Create a single continuous audiobook from structured JSON file"""
        self.logger.info(f"Creating single audiobook from: {json_path}")
        
        # Load structured text
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            raise Exception(f"Could not load JSON file: {e}")
        
        # Check for sections or chapters (backward compatibility)
        if 'sections' in data:
            sections = data['sections']
        elif 'chapters' in data:
            sections = data['chapters']  # Backward compatibility
        else:
            raise Exception("Invalid JSON format: 'sections' or 'chapters' key not found")
        
        title = data.get('title', 'Audiobook')
        
        self.logger.info(f"Title: {title}")
        self.logger.info(f"Sections to process: {len(sections)}")
        
        # Combine all sections into one continuous text
        start_time = time.time()
        full_audiobook_text = ""
        total_words = 0
        
        for i, section in enumerate(sections, 1):
            section_num = section['number']
            section_title = section['title']
            section_content = section['content']
            
            self.logger.info(f"📖 Adding Section {section_num}: {section_title}")
            
            # Add section break and title
            if i == 1:
                # First section - add book title
                full_audiobook_text += f"{title}. "
            
            # Add section intro with natural pause
            section_intro = f"Section {section_num}: {section_title}. "
            full_audiobook_text += section_intro
            
            # Add section content with natural pause at the end
            full_audiobook_text += section_content + ". "
            
            # Add longer pause between sections (except for the last one)
            if i < len(sections):
                full_audiobook_text += "... "  # Natural pause between sections
            
            total_words += section.get('word_count', len(section_content.split()))
        
        # Create safe filename for the audiobook
        import re
        safe_title = re.sub(r'[^\w\s-]', '', title)[:50].strip()
        safe_title = re.sub(r'\s+', '_', safe_title)
        
        output_filename = f"{safe_title}_complete_audiobook.wav"
        output_path = self.output_dir / output_filename
        
        self.logger.info(f"🎙️ Generating complete audiobook: {output_filename}")
        self.logger.info(f"📝 Total content length: {len(full_audiobook_text)} characters")
        self.logger.info(f"📊 Total words: {total_words:,}")
        
        # Generate the complete audiobook
        success = self.generate_audio_for_text(full_audiobook_text, str(output_path))
        
        if not success:
            raise Exception("Failed to generate audiobook audio file")
        
        # Calculate duration and create report
        end_time = time.time()
        processing_time = (end_time - start_time) / 60.0
        duration = self.get_audio_duration(str(output_path))
        
        report = {
            "title": title,
            "source_json": json_path,
            "generation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processing_time_minutes": round(processing_time, 2),
            "total_sections": len(sections),
            "total_words": total_words,
            "total_duration_minutes": round(duration, 2),
            "audiobook_file": output_filename,
            "audiobook_path": str(output_path),
            "output_directory": str(self.output_dir),
            "file_size_mb": round(os.path.getsize(output_path) / (1024 * 1024), 2) if os.path.exists(output_path) else 0
        }
        
        # Save report
        report_path = self.output_dir / "audiobook_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Print summary
        self.logger.info(f"\n{'='*60}")
        self.logger.info("🎉 COMPLETE AUDIOBOOK GENERATION FINISHED!")
        self.logger.info(f"📚 Title: {title}")
        self.logger.info(f"📖 Sections: {len(sections)}")
        self.logger.info(f"📝 Total Words: {total_words:,}")
        self.logger.info(f"⏱️  Duration: {duration:.1f} minutes")
        self.logger.info(f"💾 File Size: {report['file_size_mb']} MB")
        self.logger.info(f"🔧 Processing Time: {processing_time:.1f} minutes")
        self.logger.info(f"📁 Output: {output_path}")
        self.logger.info(f"{'='*60}")
        
        return report
    
    def create_audiobook_from_text(self, text_content: str, title: str = "Audiobook") -> Dict:
        """Create single audiobook directly from text content"""
        self.logger.info(f"Creating single audiobook from text: {title}")
        
        start_time = time.time()
        
        # Clean and prepare text
        clean_text = self.clean_text_for_tts(text_content)
        word_count = len(clean_text.split())
        
        # Create safe filename
        import re
        safe_title = re.sub(r'[^\w\s-]', '', title)[:50].strip()
        safe_title = re.sub(r'\s+', '_', safe_title)
        
        output_filename = f"{safe_title}_complete_audiobook.wav"
        output_path = self.output_dir / output_filename
        
        self.logger.info(f"🎙️ Generating complete audiobook: {output_filename}")
        self.logger.info(f"📝 Total content length: {len(clean_text)} characters")
        self.logger.info(f"📊 Total words: {word_count:,}")
        
        # Generate the complete audiobook
        success = self.generate_audio_for_text(clean_text, str(output_path))
        
        if not success:
            raise Exception("Failed to generate audiobook audio file")
        
        # Calculate duration and create report
        end_time = time.time()
        processing_time = (end_time - start_time) / 60.0
        duration = self.get_audio_duration(str(output_path))
        
        report = {
            "title": title,
            "generation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processing_time_minutes": round(processing_time, 2),
            "total_words": word_count,
            "total_duration_minutes": round(duration, 2),
            "audiobook_file": output_filename,
            "audiobook_path": str(output_path),
            "output_directory": str(self.output_dir),
            "file_size_mb": round(os.path.getsize(output_path) / (1024 * 1024), 2) if os.path.exists(output_path) else 0
        }
        
        # Save report
        report_path = self.output_dir / "audiobook_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("🎉 COMPLETE AUDIOBOOK GENERATION FINISHED!")
        self.logger.info(f"📚 Title: {title}")
        self.logger.info(f"📝 Total Words: {word_count:,}")
        self.logger.info(f"⏱️  Duration: {duration:.1f} minutes")
        self.logger.info(f"💾 File Size: {report['file_size_mb']} MB")
        self.logger.info(f"🔧 Processing Time: {processing_time:.1f} minutes")
        self.logger.info(f"📁 Output: {output_path}")
        self.logger.info(f"{'='*60}")
        
        return report
    
    def create_cloned_audiobook_from_json(self, json_file_path, output_path=None, voice_model_name=None):
        """Create audiobook using cloned voice from extracted JSON data"""
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        from voice_cloner import VoiceCloner
        
        # Load JSON data
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract content
        if isinstance(data, dict) and 'chapters' in data:
            content = ""
            for chapter in data['chapters']:
                content += f"{chapter.get('title', '')}\n\n{chapter.get('content', '')}\n\n"
        else:
            content = str(data)
        
        # Initialize enhanced voice cloner
        voice_cloner = VoiceCloner()
        
        # Use voice model if specified
        if voice_model_name:
            try:
                voice_cloner.load_voice_model(voice_model_name)
                self.logger.info(f"Loaded enhanced voice model: {voice_model_name}")
            except Exception as e:
                self.logger.warning(f"Could not load voice model {voice_model_name}: {e}")
                self.logger.info("Falling back to default voice")
        
        # Generate audiobook with cloned voice
        if output_path is None:
            json_name = Path(json_file_path).stem
            output_path = self.output_dir / "audiobooks" / f"{json_name}_cloned_voice.wav"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Creating enhanced cloned voice audiobook: {output_path}")
        
        # Generate audio using enhanced voice cloner
        try:
            audio_file = voice_cloner.generate_audiobook(content, str(output_path), voice_model_name)
            
            # Verify file was created
            if Path(audio_file).exists():
                file_size = Path(audio_file).stat().st_size
                duration = self._get_audio_duration(audio_file)
                
                self.logger.info(f"Enhanced cloned voice audiobook created successfully!")
                self.logger.info(f"File: {audio_file}")
                self.logger.info(f"Size: {file_size / (1024*1024):.2f} MB")
                self.logger.info(f"Duration: {duration:.2f} seconds")
                
                return audio_file
            else:
                raise Exception("Audio file was not created")
                
        except Exception as e:
            self.logger.error(f"Error creating enhanced cloned voice audiobook: {e}")
            raise
    
    def get_available_voice_models(self):
        """Get list of available trained voice models with enhanced information"""
        try:
            if VOICE_CLONING_AVAILABLE and self.voice_cloner:
                # Get custom trained models first (these are your voice models)
                custom_models = self.voice_cloner.get_custom_models_only()
                model_names = [model['name'] for model in custom_models if model['status'] == 'ready']
                
                # If no custom models, get all available models
                if not model_names:
                    model_names = self.voice_cloner.get_available_models()
                
                self.logger.info(f"Found {len(model_names)} voice models: {model_names}")
                return model_names
            else:
                # Return basic system voices if voice cloning not available
                if hasattr(self, 'tts_engine') and self.tts_engine:
                    try:
                        voices = self.tts_engine.getProperty('voices')
                        if voices:
                            system_voices = []
                            for voice in voices[:3]:  # Limit to first 3 system voices
                                clean_name = voice.name.replace(' ', '_').replace('Microsoft', '').strip()
                                system_voices.append(f"system_{clean_name}")
                            return system_voices
                    except:
                        pass
                return ['default', 'high_quality', 'balanced']
        except Exception as e:
            self.logger.error(f"Error getting available voice models: {e}")
            return ['default', 'system', 'fallback']
    
    def get_voice_model_info(self, model_name):
        """Get detailed information about a specific voice model"""
        try:
            if VOICE_CLONING_AVAILABLE and self.voice_cloner:
                return self.voice_cloner.get_model_info(model_name)
            else:
                # Return basic info for system voices
                return {
                    'name': model_name,
                    'model_type': 'system',
                    'characteristics': {
                        'voice_type': 'system_default',
                        'estimated_gender': 'unknown'
                    },
                    'training_quality': 0.6,
                    'description': 'System default voice'
                }
        except Exception as e:
            self.logger.error(f"Error getting model info: {e}")
            return None
    
    def get_custom_voice_models_with_info(self):
        """Get custom voice models with detailed information for display"""
        models_info = []
        try:
            if VOICE_CLONING_AVAILABLE and self.voice_cloner:
                custom_models = self.voice_cloner.get_custom_models_only()
                
                for model in custom_models:
                    model_info = self.voice_cloner.get_model_info(model['name'])
                    if model_info:
                        # Create enhanced display info
                        display_info = {
                            'name': model['name'],
                            'type': model_info.get('model_type', 'custom'),
                            'quality': model_info.get('training_quality', 0.5),
                            'samples': model_info.get('sample_count', 0),
                            'duration': model_info.get('total_duration', 0),
                            'gender': model_info.get('characteristics', {}).get('estimated_gender', 'unknown'),
                            'created': model_info.get('created_date', 'Unknown'),
                            'description': model_info.get('description', 'Custom voice model'),
                            'status': model.get('status', 'ready')
                        }
                        models_info.append(display_info)
                
                self.logger.info(f"Retrieved info for {len(models_info)} custom voice models")
                
            return models_info
            
        except Exception as e:
            self.logger.error(f"Error getting custom voice models info: {e}")
            return []
    
    def train_voice_model(self, voice_samples_dir, model_name):
        """Train a new voice model from voice samples"""
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        from voice_cloner import VoiceCloner
        
        try:
            voice_cloner = VoiceCloner()
            self.logger.info(f"Training enhanced voice model '{model_name}' from samples in {voice_samples_dir}")
            
            model_path = voice_cloner.train_voice_model(voice_samples_dir, model_name)
            
            self.logger.info(f"Enhanced voice model trained successfully: {model_path}")
            return model_path
            
        except Exception as e:
            self.logger.error(f"Error training enhanced voice model: {e}")
            raise
    
    def analyze_voice_sample(self, audio_file_path):
        """Analyze a voice sample and return characteristics"""
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        from voice_cloner import VoiceCloner
        
        try:
            voice_cloner = VoiceCloner()
            analysis = voice_cloner.analyze_voice_sample(audio_file_path)
            
            self.logger.info(f"Enhanced voice sample analysis completed for: {audio_file_path}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing voice sample: {e}")
            raise
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds (private method)"""
        try:
            if PYDUB_AVAILABLE:
                audio = AudioSegment.from_wav(audio_path)
                return len(audio) / 1000.0  # Convert to seconds
            else:
                # Fallback: estimate based on file size (very rough)
                file_size = os.path.getsize(audio_path)
                # Rough estimate: 1 MB ≈ 60 seconds for WAV
                return file_size / (1024 * 1024) * 60
        except Exception as e:
            self.logger.warning(f"Could not get audio duration for {audio_path}: {e}")
            return 0.0

def main():
    """Main function"""
    # Configuration - check if we have extracted text
    extracted_dir = Path("extracted_text")
    
    if extracted_dir.exists():
        # Look for JSON file
        json_files = list(extracted_dir.glob("*_structured.json"))
        if json_files:
            json_file = json_files[0]
            print(f"Found extracted text: {json_file}")
            
            # Create audiobook generator
            generator = AudiobookGenerator("output/audiobooks")
            
            try:
                report = generator.create_audiobook_from_json(str(json_file))
                
                print("\n✅ Single audiobook generation completed!")
                print(f"📁 Output: {generator.output_dir}")
                print(f"📖 Chapters Combined: {report['total_chapters']}")
                print(f"⏱️  Duration: {report['total_duration_minutes']:.1f} minutes")
                print(f"🎵 Audiobook File: {report['audiobook_file']}")
                print(f"💾 File Size: {report['file_size_mb']} MB")
                
            except Exception as e:
                print(f"\n❌ Error: {e}")
        else:
            print("No structured JSON files found. Run pdf_text_extractor.py first.")
    else:
        print("No extracted text found. Run pdf_text_extractor.py first.")

if __name__ == "__main__":
    main()
