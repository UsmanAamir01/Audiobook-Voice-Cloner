#!/usr/bin/env python3
"""
Simple Voice Cloner using Tortoise TTS
=====================================
A lightweight voice cloning solution that can run on laptops.
"""

import os
import logging
from pathlib import Path
import json
import shutil
import warnings
warnings.filterwarnings("ignore")

# Core dependencies
import numpy as np
import soundfile as sf

# Voice cloning
try:
    import torch
    from tortoise.api import TextToSpeech
    from tortoise.utils.audio import load_audio, get_voice_dir, get_voices
    TORTOISE_AVAILABLE = True
except ImportError:
    TORTOISE_AVAILABLE = False
    logging.warning("Tortoise TTS not available. Voice cloning will use fallback method.")

# Fallback TTS
import pyttsx3

class VoiceCloner:
    """Simple voice cloning using Tortoise TTS"""
    
    def __init__(self, voice_models_dir="voice_models", voice_samples_dir="voice_samples"):
        global TORTOISE_AVAILABLE
        
        self.voice_models_dir = Path(voice_models_dir)
        self.voice_samples_dir = Path(voice_samples_dir)
        self.voice_models_dir.mkdir(exist_ok=True)
        self.voice_samples_dir.mkdir(exist_ok=True)
        
        # Setup logging with less verbosity for production
        if not hasattr(logging.getLogger(), '_configured'):
            logging.basicConfig(
                level=logging.INFO,
                format='%(levelname)s - %(message)s',
                handlers=[logging.StreamHandler()]
            )
            logging.getLogger()._configured = True
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.WARNING)  # Reduce log verbosity
        
        # Initialize TTS engines
        self.tortoise_tts = None
        self.fallback_tts = None
        self._tts_initialized = False
        
        if TORTOISE_AVAILABLE:
            try:
                self.logger.info("Initializing Tortoise TTS for voice cloning...")
                self.tortoise_tts = TextToSpeech()
                self.logger.info("Tortoise TTS initialized successfully!")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Tortoise TTS: {e}")
                TORTOISE_AVAILABLE = False
        
        if not TORTOISE_AVAILABLE:
            self.logger.info("Using fallback TTS engine...")
            try:
                self.fallback_tts = pyttsx3.init()
                if self.fallback_tts:
                    self.fallback_tts.setProperty('rate', 200)
                    self.fallback_tts.setProperty('volume', 0.9)
                    self._tts_initialized = True
                    self.logger.info("Fallback TTS engine initialized successfully")
                else:
                    self.logger.warning("Failed to initialize fallback TTS engine")
            except Exception as e:
                self.logger.warning(f"Error initializing fallback TTS: {e}")
                self.fallback_tts = None
    
    def __del__(self):
        """Cleanup TTS engines properly"""
        try:
            if hasattr(self, 'fallback_tts') and self.fallback_tts and self._tts_initialized:
                try:
                    self.fallback_tts.stop()
                except:
                    pass
                self.fallback_tts = None
        except:
            pass
    
    def analyze_voice_sample(self, audio_file_path):
        """Analyze a voice sample and return characteristics"""
        try:
            audio_file_path = Path(audio_file_path)
            
            # For M4A files, try different approaches
            if audio_file_path.suffix.lower() == '.m4a':
                try:
                    # Try with pydub first
                    from pydub import AudioSegment
                    
                    # Load M4A file
                    audio = AudioSegment.from_file(str(audio_file_path), format="m4a")
                    
                    # Basic analysis from AudioSegment
                    duration = len(audio) / 1000.0  # Convert to seconds
                    sample_rate = audio.frame_rate
                    channels = audio.channels
                    
                    # Convert to numpy for analysis
                    audio_data = np.array(audio.get_array_of_samples())
                    if channels == 2:
                        audio_data = audio_data.reshape((-1, 2))
                        audio_data = np.mean(audio_data, axis=1)  # Convert to mono
                    
                    # Normalize to [-1, 1]
                    audio_data = audio_data.astype(np.float32) / 32768.0
                    
                    self.logger.info(f"Successfully loaded M4A file: {audio_file_path.name} ({duration:.1f}s)")
                    
                except ImportError:
                    self.logger.warning("pydub not available, cannot process M4A files")
                    return self._get_default_analysis("M4A format not supported without pydub")
                except Exception as e:
                    # Only log M4A conversion errors once, not repeatedly
                    if not hasattr(self, '_m4a_warning_shown'):
                        self.logger.warning(f"M4A file processing requires ffmpeg. Using estimation instead.")
                        self._m4a_warning_shown = True
                    
                    # Return basic info based on file size estimation
                    file_size = audio_file_path.stat().st_size
                    estimated_duration = max(5.0, file_size / 50000)  # Rough estimate
                    
                    return {
                        'duration': estimated_duration,
                        'sample_rate': 44100,  # Assumed
                        'average_volume': 0.5,  # Assumed
                        'max_volume': 0.8,  # Assumed
                        'estimated_pitch': 150.0,  # Assumed
                        'zero_crossing_rate': 75.0,  # Assumed
                        'audio_quality': 'estimated',
                        'file_format': '.m4a',
                        'channels': 'unknown',
                        'note': 'M4A file analyzed with estimation due to conversion limitations'
                    }
            else:
                # Handle WAV, MP3, and other formats with soundfile
                try:
                    audio_data, sample_rate = sf.read(str(audio_file_path))
                    duration = len(audio_data) / sample_rate
                except Exception as e:
                    self.logger.warning(f"soundfile could not read {audio_file_path}: {e}")
                    return self._get_default_analysis("Error: Could not read audio file")
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Calculate basic characteristics
            average_volume = np.mean(np.abs(audio_data))
            max_volume = np.max(np.abs(audio_data))
            
            # Simple pitch estimation (zero crossing rate)
            zero_crossings = np.where(np.diff(np.signbit(audio_data)))[0]
            zero_crossing_rate = len(zero_crossings) / duration if duration > 0 else 0
            
            # Estimate fundamental frequency (very basic)
            estimated_pitch = zero_crossing_rate / 2.0 if zero_crossing_rate > 0 else 150.0
            
            # Ensure reasonable pitch range
            if estimated_pitch < 50:
                estimated_pitch = 150.0  # Default male voice
            elif estimated_pitch > 500:
                estimated_pitch = 200.0  # Default female voice
            
            # Quality assessment
            quality = 'good'
            if average_volume < 0.001:
                quality = 'very_low'
            elif average_volume < 0.01:
                quality = 'low'
            elif duration < 3:
                quality = 'too_short'
            elif duration > 60:
                quality = 'too_long'
            
            analysis = {
                'duration': float(duration),
                'sample_rate': int(sample_rate),
                'average_volume': float(average_volume),
                'max_volume': float(max_volume),
                'estimated_pitch': float(estimated_pitch),
                'zero_crossing_rate': float(zero_crossing_rate),
                'audio_quality': quality,
                'file_format': audio_file_path.suffix.lower(),
                'channels': 'mono' if len(audio_data.shape) == 1 else 'stereo'
            }
            
            self.logger.info(f"Voice analysis completed: {duration:.1f}s, {estimated_pitch:.1f}Hz, quality: {quality}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing voice sample: {e}")
            return self._get_default_analysis(f"Error during analysis: {str(e)}")
    
    def _get_default_analysis(self, error_msg="Unknown error"):
        """Get default analysis structure for error cases"""
        return {
            'duration': 10.0,  # Default duration for estimation
            'sample_rate': 44100,
            'average_volume': 0.3,
            'max_volume': 0.7,
            'estimated_pitch': 150.0,
            'zero_crossing_rate': 75.0,
            'audio_quality': 'estimated',
            'file_format': 'unknown',
            'channels': 'unknown',
            'error': error_msg
        }
    
    def prepare_voice_samples(self, samples_dir, voice_name):
        """Prepare voice samples for Tortoise TTS"""
        if not TORTOISE_AVAILABLE:
            self.logger.warning("Tortoise TTS not available, voice samples will be processed for fallback mode")
            return self._prepare_fallback_samples(samples_dir, voice_name)
            
        try:
            samples_path = Path(samples_dir)
            if not samples_path.exists():
                self.logger.error(f"Samples directory not found: {samples_dir}")
                return False
            
            # Create voice directory in Tortoise format
            voice_dir = get_voice_dir() / voice_name
            voice_dir.mkdir(exist_ok=True)
            
            # Copy and convert audio files
            supported_formats = ['.wav', '.mp3', '.m4a', '.flac']
            copied_files = 0
            
            for audio_file in samples_path.iterdir():
                if audio_file.suffix.lower() in supported_formats:
                    try:
                        # Handle different audio formats
                        audio_data, sr = self._load_audio_safely(audio_file)
                        if audio_data is None:
                            continue
                        
                        # Convert to mono if stereo
                        if len(audio_data.shape) > 1:
                            audio_data = np.mean(audio_data, axis=1)
                        
                        # Resample to 22kHz (Tortoise preferred rate)
                        if sr != 22050:
                            try:
                                from scipy.signal import resample
                                audio_data = resample(audio_data, int(len(audio_data) * 22050 / sr))
                                sr = 22050
                            except ImportError:
                                self.logger.warning("scipy not available, keeping original sample rate")
                        
                        # Save as WAV in voice directory
                        output_file = voice_dir / f"{audio_file.stem}.wav"
                        sf.write(output_file, audio_data, sr)
                        copied_files += 1
                        
                        self.logger.info(f"Prepared sample: {output_file.name}")
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to process {audio_file.name}: {e}")
            
            if copied_files > 0:
                self.logger.info(f"Successfully prepared {copied_files} voice samples for '{voice_name}'")
                return True
            else:
                self.logger.error("No valid audio samples were processed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error preparing voice samples: {e}")
            return False
    
    def _load_audio_safely(self, audio_file):
        """Safely load audio file, handling different formats"""
        try:
            # Try soundfile first
            audio_data, sr = sf.read(str(audio_file))
            return audio_data, sr
        except Exception as e1:
            self.logger.debug(f"soundfile failed for {audio_file.name}: {e1}")
            
            # Try pydub for m4a and other formats
            try:
                from pydub import AudioSegment
                audio_segment = AudioSegment.from_file(str(audio_file))
                
                # Convert to numpy array
                audio_data = np.array(audio_segment.get_array_of_samples())
                if audio_segment.channels == 2:
                    audio_data = audio_data.reshape((-1, 2))
                
                sr = audio_segment.frame_rate
                return audio_data.astype(np.float32) / 32768.0, sr  # Normalize to [-1, 1]
                
            except Exception as e2:
                self.logger.warning(f"Could not load {audio_file.name} with soundfile or pydub: {e1}, {e2}")
                return None, None
    
    def _prepare_fallback_samples(self, samples_dir, voice_name):
        """Prepare samples for fallback mode (just verify they exist)"""
        try:
            samples_path = Path(samples_dir)
            if not samples_path.exists():
                self.logger.error(f"Samples directory not found: {samples_dir}")
                return False
            
            # Just check if we have audio files
            supported_formats = ['.wav', '.mp3', '.m4a', '.flac']
            audio_files = []
            
            for audio_file in samples_path.iterdir():
                if audio_file.suffix.lower() in supported_formats:
                    audio_files.append(audio_file)
            
            if audio_files:
                self.logger.info(f"Found {len(audio_files)} audio samples for fallback mode")
                return True
            else:
                self.logger.error("No valid audio samples found")
                return False
                
        except Exception as e:
            self.logger.error(f"Error checking fallback samples: {e}")
            return False
    
    def train_voice_model(self, samples_dir, model_name):
        """Train voice model (prepare samples for Tortoise TTS)"""
        try:
            self.logger.info(f"Training voice model '{model_name}'...")
            
            # Analyze samples for model info
            samples_path = Path(samples_dir)
            sample_files = []
            for ext in ['.wav', '.mp3', '.m4a', '.flac']:
                sample_files.extend(samples_path.glob(f"*{ext}"))
            
            # Analyze first sample for characteristics
            characteristics = {}
            total_duration = 0
            
            if sample_files:
                try:
                    first_sample_analysis = self.analyze_voice_sample(str(sample_files[0]))
                    characteristics = {
                        'estimated_gender': self._estimate_gender(first_sample_analysis.get('estimated_pitch', 150)),
                        'average_pitch': first_sample_analysis.get('estimated_pitch', 150),
                        'audio_quality': first_sample_analysis.get('audio_quality', 'unknown'),
                        'voice_type': 'custom_trained'
                    }
                    
                    # Calculate total duration from all samples
                    for sample_file in sample_files:
                        try:
                            sample_analysis = self.analyze_voice_sample(str(sample_file))
                            total_duration += sample_analysis.get('duration', 0)
                        except:
                            continue
                            
                except Exception as e:
                    self.logger.warning(f"Could not analyze samples: {e}")
            
            if TORTOISE_AVAILABLE:
                # Prepare samples for Tortoise TTS
                success = self.prepare_voice_samples(samples_dir, model_name)
                if success:
                    # Save enhanced model info
                    model_info = {
                        'name': model_name,
                        'type': 'tortoise_tts',
                        'samples_dir': str(samples_dir),
                        'created_date': str(Path().cwd()),
                        'status': 'ready',
                        'sample_count': len(sample_files),
                        'total_duration': total_duration,
                        'characteristics': characteristics,
                        'training_quality': min(1.0, max(0.1, total_duration / 60.0)),  # Quality based on duration
                        'model_type': 'custom',
                        'description': f"Custom voice model trained from {len(sample_files)} samples"
                    }
                    
                    model_file = self.voice_models_dir / f"{model_name}.json"
                    with open(model_file, 'w', encoding='utf-8') as f:
                        json.dump(model_info, f, indent=2, ensure_ascii=False)
                    
                    self.logger.info(f"Voice model '{model_name}' trained successfully with {len(sample_files)} samples!")
                    return str(model_file)
                else:
                    raise Exception("Failed to prepare voice samples")
            else:
                # Fallback: save enhanced sample info
                model_info = {
                    'name': model_name,
                    'type': 'fallback',
                    'samples_dir': str(samples_dir),
                    'created_date': str(Path().cwd()),
                    'status': 'ready',
                    'sample_count': len(sample_files),
                    'total_duration': total_duration,
                    'characteristics': characteristics,
                    'training_quality': min(1.0, max(0.1, total_duration / 60.0)),
                    'model_type': 'custom',
                    'description': f"Custom voice model (fallback mode) from {len(sample_files)} samples"
                }
                
                model_file = self.voice_models_dir / f"{model_name}.json"
                with open(model_file, 'w', encoding='utf-8') as f:
                    json.dump(model_info, f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"Voice model '{model_name}' saved with {len(sample_files)} samples (using fallback TTS)")
                return str(model_file)
                
        except Exception as e:
            self.logger.error(f"Error training voice model: {e}")
            raise
    
    def get_available_models(self):
        """Get list of available voice models"""
        models = []
        
        try:
            # Check local trained models first (these are your custom voice models)
            if self.voice_models_dir.exists():
                for model_file in self.voice_models_dir.glob("*.json"):
                    try:
                        with open(model_file, 'r', encoding='utf-8') as f:
                            model_info = json.load(f)
                        model_name = model_info.get('name', model_file.stem)
                        if model_name not in models:
                            models.append(model_name)
                        self.logger.debug(f"Found custom voice model: {model_name}")
                    except Exception as e:
                        self.logger.warning(f"Could not load model file {model_file}: {e}")
                        continue
            
            # Check Tortoise voices if available (pre-trained voices)
            if TORTOISE_AVAILABLE:
                try:
                    tortoise_voices = get_voices()
                    for voice in tortoise_voices:
                        voice_name = f"tortoise_{voice}"
                        if voice_name not in models:
                            models.append(voice_name)
                        self.logger.debug(f"Found Tortoise voice: {voice_name}")
                except Exception as e:
                    self.logger.warning(f"Could not load Tortoise voices: {e}")
            
            # Only add default voices if no custom models are available
            if not models:
                system_voices = ['default', 'system', 'fallback']
                models.extend(system_voices)
            
            self.logger.info(f"Available voice models: {models}")
            return models
            
        except Exception as e:
            self.logger.error(f"Error getting available models: {e}")
            return ['default', 'system', 'fallback']
    
    def get_model_info(self, model_name):
        """Get detailed information about a voice model"""
        try:
            model_file = self.voice_models_dir / f"{model_name}.json"
            
            if model_file.exists():
                # Load custom model info
                with open(model_file, 'r', encoding='utf-8') as f:
                    model_info = json.load(f)
                
                # Enhance with sample analysis if available
                samples_dir = model_info.get('samples_dir')
                if samples_dir and Path(samples_dir).exists():
                    # Analyze samples to get characteristics
                    sample_files = []
                    characteristics = {}
                    total_duration = 0
                    
                    for ext in ['.wav', '.mp3', '.m4a', '.flac']:
                        sample_files.extend(Path(samples_dir).glob(f"*{ext}"))
                    
                    if sample_files:
                        # Analyze first sample for characteristics
                        try:
                            first_sample_analysis = self.analyze_voice_sample(str(sample_files[0]))
                            characteristics = {
                                'estimated_gender': self._estimate_gender(first_sample_analysis.get('estimated_pitch', 150)),
                                'pitch': {
                                    'mean': first_sample_analysis.get('estimated_pitch', 150),
                                    'quality': first_sample_analysis.get('audio_quality', 'unknown')
                                },
                                'voice_type': 'custom_trained'
                            }
                            
                            # Calculate total duration from all samples
                            for sample_file in sample_files:
                                try:
                                    sample_analysis = self.analyze_voice_sample(str(sample_file))
                                    total_duration += sample_analysis.get('duration', 0)
                                except:
                                    continue
                                    
                        except Exception as e:
                            self.logger.debug(f"Could not analyze samples: {e}")
                    
                    model_info.update({
                        'sample_count': len(sample_files),
                        'total_duration': total_duration,
                        'characteristics': characteristics,
                        'training_quality': min(1.0, total_duration / 60.0),  # Quality based on duration
                        'model_type': 'custom',
                        'created_date': model_info.get('created_date', 'Unknown')
                    })
                
                return model_info
            
            elif model_name.startswith('tortoise_'):
                # Tortoise pre-trained model
                return {
                    'name': model_name,
                    'model_type': 'tortoise_pretrained',
                    'characteristics': {
                        'voice_type': 'pre_trained',
                        'estimated_gender': 'unknown',
                        'pitch': {'mean': 0, 'quality': 'high'}
                    },
                    'training_quality': 0.8,
                    'sample_count': 'multiple',
                    'created_date': 'Pre-trained',
                    'description': 'High-quality pre-trained voice from Tortoise TTS'
                }
            
            else:
                # System/default voice
                return {
                    'name': model_name,
                    'model_type': 'system',
                    'characteristics': {
                        'voice_type': 'system_default',
                        'estimated_gender': 'unknown',
                        'pitch': {'mean': 0, 'quality': 'standard'}
                    },
                    'training_quality': 0.6,
                    'sample_count': 'system',
                    'created_date': 'Built-in',
                    'description': 'System default voice'
                }
                
        except Exception as e:
            self.logger.error(f"Error getting model info for {model_name}: {e}")
            return None
    
    def _estimate_gender(self, pitch):
        """Simple gender estimation based on pitch"""
        if pitch > 180:
            return 'female'
        elif pitch > 120:
            return 'male'
        else:
            return 'unknown'
    
    def get_custom_models_only(self):
        """Get only custom trained models (your voice models)"""
        custom_models = []
        
        try:
            if self.voice_models_dir.exists():
                for model_file in self.voice_models_dir.glob("*.json"):
                    try:
                        with open(model_file, 'r', encoding='utf-8') as f:
                            model_info = json.load(f)
                        model_name = model_info.get('name', model_file.stem)
                        model_type = model_info.get('type', 'unknown')
                        
                        # Only include custom trained models
                        if model_type in ['tortoise_tts', 'fallback'] and model_name:
                            custom_models.append({
                                'name': model_name,
                                'type': model_type,
                                'status': model_info.get('status', 'unknown'),
                                'created_date': model_info.get('created_date', 'Unknown')
                            })
                            
                    except Exception as e:
                        self.logger.warning(f"Could not load model file {model_file}: {e}")
                        continue
            
            self.logger.info(f"Found {len(custom_models)} custom voice models")
            return custom_models
            
        except Exception as e:
            self.logger.error(f"Error getting custom models: {e}")
    def upgrade_existing_models(self):
        """Upgrade existing voice models with enhanced information"""
        try:
            upgraded_count = 0
            
            if self.voice_models_dir.exists():
                for model_file in self.voice_models_dir.glob("*.json"):
                    try:
                        with open(model_file, 'r', encoding='utf-8') as f:
                            model_info = json.load(f)
                        
                        # Check if model needs upgrading
                        if 'characteristics' not in model_info or 'training_quality' not in model_info:
                            self.logger.info(f"Upgrading model: {model_info.get('name', model_file.stem)}")
                            
                            # Analyze samples for enhanced info
                            samples_dir = model_info.get('samples_dir')
                            if samples_dir and Path(samples_dir).exists():
                                sample_files = []
                                for ext in ['.wav', '.mp3', '.m4a', '.flac']:
                                    sample_files.extend(Path(samples_dir).glob(f"*{ext}"))
                                
                                characteristics = {}
                                total_duration = 0
                                
                                if sample_files:
                                    try:
                                        # Analyze first sample
                                        first_sample_analysis = self.analyze_voice_sample(str(sample_files[0]))
                                        characteristics = {
                                            'estimated_gender': self._estimate_gender(first_sample_analysis.get('estimated_pitch', 150)),
                                            'average_pitch': first_sample_analysis.get('estimated_pitch', 150),
                                            'audio_quality': first_sample_analysis.get('audio_quality', 'good'),
                                            'voice_type': 'custom_trained'
                                        }
                                        
                                        # Calculate total duration
                                        for sample_file in sample_files:
                                            try:
                                                sample_analysis = self.analyze_voice_sample(str(sample_file))
                                                total_duration += sample_analysis.get('duration', 0)
                                            except:
                                                continue
                                                
                                    except Exception as e:
                                        self.logger.warning(f"Could not analyze samples for {model_info.get('name')}: {e}")
                                        characteristics = {
                                            'estimated_gender': 'unknown',
                                            'average_pitch': 150,
                                            'audio_quality': 'unknown',
                                            'voice_type': 'custom_trained'
                                        }
                                
                                # Update model info
                                model_info.update({
                                    'sample_count': len(sample_files),
                                    'total_duration': total_duration,
                                    'characteristics': characteristics,
                                    'training_quality': min(1.0, max(0.1, total_duration / 60.0)),
                                    'model_type': 'custom',
                                    'description': f"Custom voice model trained from {len(sample_files)} samples",
                                    'upgraded': True
                                })
                                
                                # Save upgraded model
                                with open(model_file, 'w', encoding='utf-8') as f:
                                    json.dump(model_info, f, indent=2, ensure_ascii=False)
                                
                                upgraded_count += 1
                                self.logger.info(f"Successfully upgraded model: {model_info.get('name')}")
                            
                    except Exception as e:
                        self.logger.warning(f"Failed to upgrade model {model_file}: {e}")
                        continue
            
            if upgraded_count > 0:
                self.logger.info(f"Upgraded {upgraded_count} voice models with enhanced information")
                return upgraded_count
            else:
                self.logger.info("All voice models are already up to date")
                return 0
                
        except Exception as e:
            self.logger.error(f"Error upgrading existing models: {e}")
            return 0
    
    def load_voice_model(self, model_name):
        """Load a voice model"""
        try:
            model_file = self.voice_models_dir / f"{model_name}.json"
            if model_file.exists():
                with open(model_file, 'r') as f:
                    model_info = json.load(f)
                self.logger.info(f"Loaded voice model: {model_name}")
                return model_info
            else:
                self.logger.warning(f"Voice model not found: {model_name}")
                return None
        except Exception as e:
            self.logger.error(f"Error loading voice model: {e}")
            return None
    
    def generate_speech(self, text, voice_model=None, output_path=None):
        """Generate speech using the specified voice model"""
        try:
            if TORTOISE_AVAILABLE and voice_model and voice_model not in ['default', 'system', 'fallback']:
                # Use Tortoise TTS for voice cloning
                self.logger.info(f"Generating speech with Tortoise TTS using voice: {voice_model}")
                
                # Remove 'tortoise_' prefix if present
                tortoise_voice = voice_model.replace('tortoise_', '')
                
                try:
                    # Generate speech
                    gen = self.tortoise_tts.tts_with_preset(
                        text, 
                        voice_samples=None,
                        preset='fast',  # Use 'standard' or 'high_quality' for better results
                        voice=tortoise_voice
                    )
                    
                    # Convert to numpy array
                    audio_data = gen.squeeze().cpu().numpy()
                    
                    # Save audio
                    if output_path:
                        sf.write(output_path, audio_data, 24000)  # Tortoise output rate
                        self.logger.info(f"Generated speech saved to: {output_path}")
                        return output_path
                    else:
                        return audio_data
                        
                except Exception as e:
                    self.logger.warning(f"Tortoise TTS failed: {e}, falling back to system TTS")
                    # Fall through to fallback TTS
            
            # Use fallback TTS (pyttsx3)
            self.logger.info(f"Generating speech with fallback TTS{' (voice: ' + str(voice_model) + ')' if voice_model else ''}")
            
            # Initialize fallback TTS if needed
            if self.fallback_tts is None:
                self.fallback_tts = pyttsx3.init()
                self.fallback_tts.setProperty('rate', 200)
                self.fallback_tts.setProperty('volume', 0.9)
            
            # Apply voice model if it's a system voice
            if voice_model and voice_model not in ['default', 'fallback']:
                try:
                    voices = self.fallback_tts.getProperty('voices')
                    if voices:
                        # Try to find matching voice
                        for voice in voices:
                            if voice_model.lower() in voice.name.lower():
                                self.fallback_tts.setProperty('voice', voice.id)
                                self.logger.info(f"Applied system voice: {voice.name}")
                                break
                except Exception as e:
                    self.logger.warning(f"Could not apply voice {voice_model}: {e}")
            
            if output_path:
                self.fallback_tts.save_to_file(text, output_path)
                self.fallback_tts.runAndWait()
                
                # Verify file was created
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    self.logger.info(f"Generated speech saved to: {output_path}")
                    return output_path
                else:
                    raise Exception("Audio file was not created or is empty")
            else:
                # For fallback, we need to save to a temp file first
                temp_file = "temp_speech.wav"
                self.fallback_tts.save_to_file(text, temp_file)
                self.fallback_tts.runAndWait()
                
                if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                    audio_data, sr = sf.read(temp_file)
                    os.remove(temp_file)
                    return audio_data
                else:
                    raise Exception("Temporary audio file was not created")
                    
        except Exception as e:
            self.logger.error(f"Error generating speech: {e}")
            raise
    
    def generate_audiobook(self, text, output_path, voice_model=None):
        """Generate a complete audiobook using voice cloning"""
        try:
            self.logger.info(f"Generating audiobook with voice model: {voice_model or 'default'}")
            
            # Split text into chunks for better processing
            max_chunk_size = 500  # words
            words = text.split()
            
            audio_chunks = []
            
            for i in range(0, len(words), max_chunk_size):
                chunk_words = words[i:i + max_chunk_size]
                chunk_text = ' '.join(chunk_words)
                
                self.logger.info(f"Processing chunk {i//max_chunk_size + 1}/{(len(words)-1)//max_chunk_size + 1}")
                
                # Generate audio for this chunk
                audio_data = self.generate_speech(chunk_text, voice_model)
                
                if isinstance(audio_data, str):
                    # If it's a file path, read the audio
                    audio_data, _ = sf.read(audio_data)
                
                audio_chunks.append(audio_data)
            
            # Combine all chunks
            if audio_chunks:
                combined_audio = np.concatenate(audio_chunks)
                
                # Save combined audiobook
                sf.write(output_path, combined_audio, 22050)
                self.logger.info(f"Audiobook generated successfully: {output_path}")
                return output_path
            else:
                raise Exception("No audio chunks were generated")
                
        except Exception as e:
            self.logger.error(f"Error generating audiobook: {e}")
            raise

def main():
    """Test the voice cloner"""
    cloner = VoiceCloner()
    
    # Test voice analysis
    print("Available models:", cloner.get_available_models())
    
    # Test speech generation
    test_text = "Hello, this is a test of the voice cloning system."
    output_file = "test_output.wav"
    
    result = cloner.generate_speech(test_text, output_path=output_file)
    print(f"Generated speech: {result}")

if __name__ == "__main__":
    main()
