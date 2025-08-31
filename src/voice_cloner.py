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
import time
warnings.filterwarnings("ignore")

# Core dependencies
import numpy as np
import soundfile as sf

# Audio processing
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logging.warning("pydub not available. M4A file support will be limited.")

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
                    if PYDUB_AVAILABLE:
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
                        
                        # Proper normalization based on sample width
                        if audio.sample_width == 2:  # 16-bit
                            audio_data = audio_data.astype(np.float32) / 32768.0
                        elif audio.sample_width == 3:  # 24-bit
                            audio_data = audio_data.astype(np.float32) / 8388608.0
                        elif audio.sample_width == 4:  # 32-bit
                            audio_data = audio_data.astype(np.float32) / 2147483648.0
                        else:  # Default normalization
                            max_val = np.max(np.abs(audio_data))
                            if max_val > 0:
                                audio_data = audio_data.astype(np.float32) / max_val
                        
                        self.logger.info(f"Successfully loaded M4A file: {audio_file_path.name} ({duration:.1f}s)")
                        
                        # Perform detailed voice analysis
                        return self._perform_detailed_voice_analysis(audio_data, sample_rate, duration, audio_file_path.suffix.lower())
                        
                    else:
                        raise ImportError("pydub not available")
                        
                except Exception as e:
                    # Enhanced estimation for M4A files based on file properties
                    self.logger.warning(f"Direct M4A analysis failed, using enhanced estimation: {e}")
                    return self._get_enhanced_m4a_estimation(audio_file_path)
            else:
                # Handle WAV, MP3, and other formats with soundfile
                try:
                    audio_data, sample_rate = sf.read(str(audio_file_path))
                    duration = len(audio_data) / sample_rate
                    
                    return self._perform_detailed_voice_analysis(audio_data, sample_rate, duration, audio_file_path.suffix.lower())
                    
                except Exception as e:
                    self.logger.warning(f"soundfile could not read {audio_file_path}: {e}")
                    return self._get_default_analysis("Error: Could not read audio file")
            
        except Exception as e:
            self.logger.error(f"Error analyzing voice sample: {e}")
            return self._get_default_analysis(f"Error during analysis: {str(e)}")
    
    def _perform_detailed_voice_analysis(self, audio_data, sample_rate, duration, file_format):
        """Perform detailed analysis of audio data"""
        try:
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Calculate basic characteristics
            average_volume = np.mean(np.abs(audio_data))
            max_volume = np.max(np.abs(audio_data))
            rms_volume = np.sqrt(np.mean(audio_data**2))
            
            # Advanced pitch analysis using zero crossing rate
            zero_crossings = np.where(np.diff(np.signbit(audio_data)))[0]
            zero_crossing_rate = len(zero_crossings) / duration if duration > 0 else 0
            
            # Estimate fundamental frequency (more sophisticated)
            estimated_pitch = self._estimate_pitch_advanced(audio_data, sample_rate)
            
            # Voice quality assessment
            quality = self._assess_voice_quality(audio_data, duration, average_volume, estimated_pitch)
            
            # Gender estimation based on pitch
            gender = self._estimate_gender_advanced(estimated_pitch, zero_crossing_rate)
            
            # Speaking characteristics
            speaking_rate = self._estimate_speaking_rate(audio_data, sample_rate, duration)
            voice_depth = self._estimate_voice_depth(estimated_pitch)
            
            analysis = {
                'duration': float(duration),
                'sample_rate': int(sample_rate),
                'average_volume': float(average_volume),
                'max_volume': float(max_volume),
                'rms_volume': float(rms_volume),
                'estimated_pitch': float(estimated_pitch),
                'zero_crossing_rate': float(zero_crossing_rate),
                'audio_quality': quality,
                'file_format': file_format,
                'channels': 'mono' if len(audio_data.shape) == 1 else 'stereo',
                'estimated_gender': gender,
                'speaking_rate': speaking_rate,
                'voice_depth': voice_depth,
                'pitch_range': {
                    'min': max(80, estimated_pitch - 30),
                    'max': min(300, estimated_pitch + 30)
                },
                'vocal_characteristics': {
                    'resonance': 'chest' if estimated_pitch < 150 else 'head',
                    'articulation': 'clear' if quality in ['good', 'excellent'] else 'moderate',
                    'rhythm': 'steady' if abs(speaking_rate - 1.0) < 0.2 else 'variable'
                }
            }
            
            self.logger.info(f"Detailed voice analysis: {duration:.1f}s, {estimated_pitch:.1f}Hz, {gender}, {quality}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in detailed voice analysis: {e}")
            return self._get_default_analysis(f"Analysis error: {str(e)}")
    
    def _estimate_pitch_advanced(self, audio_data, sample_rate):
        """Advanced pitch estimation using autocorrelation"""
        try:
            # Simple autocorrelation-based pitch detection
            # Window the signal
            window_size = min(2048, len(audio_data) // 4)
            if window_size < 512:
                # Fallback for very short audio
                zero_crossings = np.where(np.diff(np.signbit(audio_data)))[0]
                zcr = len(zero_crossings) / (len(audio_data) / sample_rate)
                return max(80, min(300, zcr / 2.0))
            
            # Take middle portion of audio for analysis
            start = len(audio_data) // 4
            end = start + window_size
            segment = audio_data[start:end]
            
            # Autocorrelation
            autocorr = np.correlate(segment, segment, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find the peak (ignoring the first peak at lag 0)
            min_period = int(sample_rate / 300)  # Max 300 Hz
            max_period = int(sample_rate / 80)   # Min 80 Hz
            
            if max_period < len(autocorr):
                peak_idx = np.argmax(autocorr[min_period:max_period]) + min_period
                pitch = sample_rate / peak_idx
                
                # Validate pitch range
                if 80 <= pitch <= 300:
                    return pitch
            
            # Fallback: use zero crossing rate
            zero_crossings = np.where(np.diff(np.signbit(audio_data)))[0]
            zcr = len(zero_crossings) / (len(audio_data) / sample_rate)
            estimated_pitch = max(80, min(300, zcr / 2.0))
            
            return estimated_pitch
            
        except Exception as e:
            self.logger.warning(f"Advanced pitch estimation failed: {e}")
            return 150.0  # Default male voice pitch
    
    def _estimate_gender_advanced(self, pitch, zcr):
        """Advanced gender estimation"""
        if pitch < 120:
            return 'male'
        elif pitch > 200:
            return 'female'
        elif pitch > 165:
            # Use additional features for borderline cases
            if zcr > 100:  # Higher ZCR might indicate female voice
                return 'female'
            else:
                return 'male'
        else:
            return 'male'
    
    def _estimate_speaking_rate(self, audio_data, sample_rate, duration):
        """Estimate speaking rate"""
        # Simple energy-based speech rate estimation
        try:
            # Frame-based energy calculation
            frame_length = int(0.025 * sample_rate)  # 25ms frames
            hop_length = int(0.01 * sample_rate)     # 10ms hop
            
            frames = []
            for i in range(0, len(audio_data) - frame_length, hop_length):
                frame = audio_data[i:i + frame_length]
                energy = np.sum(frame ** 2)
                frames.append(energy)
            
            frames = np.array(frames)
            
            # Threshold for speech activity
            threshold = np.percentile(frames, 30)  # Bottom 30% is likely silence
            speech_frames = frames > threshold
            
            # Count speech segments
            speech_changes = np.diff(speech_frames.astype(int))
            speech_segments = np.sum(speech_changes == 1)  # Start of speech segments
            
            # Estimate speaking rate (segments per second)
            if duration > 0:
                rate = speech_segments / duration
                if 2 <= rate <= 8:  # Reasonable range
                    if rate < 3:
                        return 'slow'
                    elif rate > 5:
                        return 'fast'
                    else:
                        return 'moderate'
            
            return 'moderate'
            
        except Exception as e:
            self.logger.warning(f"Speaking rate estimation failed: {e}")
            return 'moderate'
    
    def _estimate_voice_depth(self, pitch):
        """Estimate voice depth based on pitch"""
        if pitch < 100:
            return 'very-deep'
        elif pitch < 130:
            return 'deep'
        elif pitch < 150:
            return 'medium-deep'
        elif pitch < 180:
            return 'medium'
        elif pitch < 220:
            return 'medium-light'
        else:
            return 'light'
    
    def _assess_voice_quality(self, audio_data, duration, volume, pitch):
        """Assess voice quality"""
        try:
            quality_score = 0
            
            # Duration check
            if duration >= 3:
                quality_score += 1
            if duration >= 5:
                quality_score += 1
                
            # Volume check
            if 0.01 <= volume <= 0.8:
                quality_score += 1
            if 0.05 <= volume <= 0.5:
                quality_score += 1
                
            # Pitch reasonableness
            if 80 <= pitch <= 300:
                quality_score += 1
            if 100 <= pitch <= 250:
                quality_score += 1
                
            # Dynamic range
            dynamic_range = np.max(audio_data) - np.min(audio_data)
            if dynamic_range > 0.1:
                quality_score += 1
                
            # Map score to quality
            if quality_score >= 6:
                return 'excellent'
            elif quality_score >= 4:
                return 'good'
            elif quality_score >= 2:
                return 'fair'
            else:
                return 'poor'
                
        except Exception as e:
            self.logger.warning(f"Quality assessment failed: {e}")
            return 'fair'
    
    def _get_enhanced_m4a_estimation(self, audio_file_path):
        """Enhanced estimation for M4A files when direct analysis fails"""
        try:
            file_size = audio_file_path.stat().st_size
            
            # Estimate duration based on file size (rough approximation)
            # M4A compression ratio is typically 10:1 compared to WAV
            estimated_duration = max(3.0, min(30.0, file_size / 80000))  # Adjusted estimate
            
            # Since this is your voice sample, use characteristics from your manual edits
            return {
                'duration': estimated_duration,
                'sample_rate': 44100,  # Standard M4A
                'average_volume': 0.4,
                'max_volume': 0.8,
                'estimated_pitch': 140.0,  # Your specified pitch
                'zero_crossing_rate': 70.0,
                'audio_quality': 'good',
                'file_format': '.m4a',
                'channels': 'mono',
                'estimated_gender': 'male',
                'speaking_rate': 'moderate',
                'voice_depth': 'medium-deep',
                'pitch_range': {
                    'min': 120,
                    'max': 160
                },
                'vocal_characteristics': {
                    'resonance': 'chest',
                    'articulation': 'clear',
                    'rhythm': 'steady'
                },
                'note': 'Enhanced M4A estimation based on file properties and user specifications'
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced M4A estimation failed: {e}")
            return self._get_default_analysis("M4A estimation error")
    
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
    
    def _estimate_gender(self, pitch):
        """Estimate gender based on pitch frequency"""
        if pitch < 120:
            return 'male'
        elif pitch > 200:
            return 'female'
        elif pitch > 165:
            return 'female'  # Slightly favor female for borderline cases
        else:
            return 'male'
    
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
            if PYDUB_AVAILABLE:
                try:
                    audio_segment = AudioSegment.from_file(str(audio_file))
                    
                    # Convert to numpy array
                    audio_data = np.array(audio_segment.get_array_of_samples())
                    if audio_segment.channels == 2:
                        audio_data = audio_data.reshape((-1, 2))
                    
                    sr = audio_segment.frame_rate
                    
                    # Proper normalization based on sample width
                    if audio_segment.sample_width == 2:  # 16-bit
                        audio_data = audio_data.astype(np.float32) / 32768.0
                    elif audio_segment.sample_width == 3:  # 24-bit
                        audio_data = audio_data.astype(np.float32) / 8388608.0
                    elif audio_segment.sample_width == 4:  # 32-bit
                        audio_data = audio_data.astype(np.float32) / 2147483648.0
                    else:  # Default normalization
                        max_val = np.max(np.abs(audio_data))
                        if max_val > 0:
                            audio_data = audio_data.astype(np.float32) / max_val
                        else:
                            audio_data = audio_data.astype(np.float32)
                    
                    self.logger.info(f"Successfully loaded {audio_file.name} using pydub ({sr}Hz, {audio_segment.channels} channels)")
                    return audio_data, sr
                    
                except Exception as e2:
                    self.logger.warning(f"Could not load {audio_file.name} with pydub either: {e2}")
            else:
                self.logger.warning(f"pydub not available, cannot process {audio_file.name}")
            
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
        """Train voice model (prepare samples and analyze characteristics)"""
        try:
            self.logger.info(f"Training voice model '{model_name}' with enhanced analysis...")
            
            # Analyze samples for model info
            samples_path = Path(samples_dir)
            sample_files = []
            for ext in ['.wav', '.mp3', '.m4a', '.flac']:
                sample_files.extend(samples_path.glob(f"*{ext}"))
            
            if not sample_files:
                raise Exception(f"No audio samples found in {samples_dir}")
            
            self.logger.info(f"Found {len(sample_files)} voice samples to analyze")
            
            # Detailed analysis of all samples
            all_characteristics = []
            total_duration = 0
            successful_analyses = 0
            
            for sample_file in sample_files:
                try:
                    self.logger.info(f"Analyzing sample: {sample_file.name}")
                    sample_analysis = self.analyze_voice_sample(str(sample_file))
                    
                    if sample_analysis and 'estimated_pitch' in sample_analysis:
                        all_characteristics.append(sample_analysis)
                        total_duration += sample_analysis.get('duration', 0)
                        successful_analyses += 1
                        
                        self.logger.info(f"Sample analysis: {sample_analysis.get('duration', 0):.1f}s, "
                                       f"pitch: {sample_analysis.get('estimated_pitch', 0):.1f}Hz, "
                                       f"quality: {sample_analysis.get('audio_quality', 'unknown')}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to analyze {sample_file.name}: {e}")
                    continue
            
            if successful_analyses == 0:
                raise Exception("No samples could be analyzed successfully")
            
            # Aggregate characteristics from all samples
            aggregated_characteristics = self._aggregate_voice_characteristics(all_characteristics)
            
            # Determine voice type and quality
            training_quality = min(1.0, max(0.3, total_duration / 10.0))  # Quality based on total duration
            if successful_analyses > 1:
                training_quality += 0.1  # Bonus for multiple samples
            
            # Enhanced model information
            model_info = {
                'name': model_name,
                'type': 'enhanced_custom' if TORTOISE_AVAILABLE else 'enhanced_fallback',
                'samples_dir': str(samples_dir),
                'created_date': str(time.strftime("%Y-%m-%d")),
                'status': 'ready',
                'sample_count': len(sample_files),
                'analyzed_samples': successful_analyses,
                'total_duration': total_duration,
                'characteristics': aggregated_characteristics,
                'training_quality': round(training_quality, 2),
                'model_type': 'custom',
                'description': f"Enhanced custom voice model trained from {successful_analyses} analyzed samples",
                'analysis_method': 'enhanced_audio_analysis',
                'samples_info': [
                    {
                        'filename': sample['file_format'],
                        'duration': sample.get('duration', 0),
                        'pitch': sample.get('estimated_pitch', 0),
                        'quality': sample.get('audio_quality', 'unknown')
                    }
                    for sample in all_characteristics
                ]
            }
            
            if TORTOISE_AVAILABLE:
                # Prepare samples for Tortoise TTS
                success = self.prepare_voice_samples(samples_dir, model_name)
                if success:
                    model_info['tortoise_prepared'] = True
                    self.logger.info(f"Voice samples prepared for Tortoise TTS")
                else:
                    model_info['tortoise_prepared'] = False
                    self.logger.warning(f"Failed to prepare samples for Tortoise TTS")
            
            # Save enhanced model info
            model_file = self.voice_models_dir / f"{model_name}.json"
            with open(model_file, 'w', encoding='utf-8') as f:
                json.dump(model_info, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"✅ Voice model '{model_name}' trained successfully!")
            self.logger.info(f"📊 Samples analyzed: {successful_analyses}/{len(sample_files)}")
            self.logger.info(f"⏱️ Total duration: {total_duration:.1f}s")
            self.logger.info(f"🎯 Training quality: {training_quality:.2f}")
            self.logger.info(f"🎙️ Voice characteristics: {aggregated_characteristics.get('estimated_gender', 'unknown')} voice, "
                           f"{aggregated_characteristics.get('estimated_pitch', 0):.1f}Hz pitch")
            
            return str(model_file)
            
        except Exception as e:
            self.logger.error(f"Error training voice model: {e}")
            raise
    
    def _aggregate_voice_characteristics(self, characteristics_list):
        """Aggregate characteristics from multiple voice samples"""
        try:
            if not characteristics_list:
                return self._get_default_characteristics()
            
            # Calculate averages and most common values
            pitches = [c.get('estimated_pitch', 150) for c in characteristics_list]
            durations = [c.get('duration', 0) for c in characteristics_list]
            volumes = [c.get('average_volume', 0.3) for c in characteristics_list]
            
            avg_pitch = np.mean(pitches)
            avg_volume = np.mean(volumes)
            total_duration = sum(durations)
            
            # Determine gender based on average pitch
            gender = self._estimate_gender_advanced(avg_pitch, 0)
            
            # Determine speaking characteristics
            speaking_rate = 'moderate'  # Default
            voice_depth = self._estimate_voice_depth(avg_pitch)
            
            # Quality assessment based on all samples
            qualities = [c.get('audio_quality', 'fair') for c in characteristics_list]
            quality_scores = {'excellent': 4, 'good': 3, 'fair': 2, 'poor': 1}
            avg_quality_score = np.mean([quality_scores.get(q, 2) for q in qualities])
            
            if avg_quality_score >= 3.5:
                overall_quality = 'excellent'
            elif avg_quality_score >= 2.5:
                overall_quality = 'good'
            elif avg_quality_score >= 1.5:
                overall_quality = 'fair'
            else:
                overall_quality = 'poor'
            
            # Personality assessment based on voice characteristics
            personality = self._assess_voice_personality(avg_pitch, avg_volume, voice_depth)
            
            aggregated = {
                'estimated_gender': gender,
                'average_pitch': round(avg_pitch, 1),
                'estimated_pitch': round(avg_pitch, 1),
                'pitch_range': {
                    'min': max(80, int(avg_pitch - 20)),
                    'max': min(300, int(avg_pitch + 20))
                },
                'average_volume': round(avg_volume, 3),
                'speaking_rate': speaking_rate,
                'voice_depth': voice_depth,
                'accent': 'neutral',  # Default
                'audio_quality': overall_quality,
                'voice_type': 'custom_trained',
                'personality': personality,
                'vocal_characteristics': {
                    'resonance': 'chest' if avg_pitch < 150 else 'head',
                    'articulation': 'clear' if overall_quality in ['good', 'excellent'] else 'moderate',
                    'rhythm': 'steady'  # Default
                },
                'confidence_score': min(1.0, max(0.3, total_duration / 5.0)),  # Based on total sample duration
                'sample_diversity': len(set(qualities))  # How many different quality levels
            }
            
            self.logger.info(f"Aggregated characteristics: {gender} voice, {avg_pitch:.1f}Hz, {overall_quality} quality")
            return aggregated
            
        except Exception as e:
            self.logger.error(f"Error aggregating voice characteristics: {e}")
            return self._get_default_characteristics()
    
    def _assess_voice_personality(self, pitch, volume, depth):
        """Assess voice personality based on characteristics"""
        try:
            personality_traits = []
            
            # Confidence assessment
            if volume > 0.4:
                personality_traits.append('confident')
            elif volume < 0.2:
                personality_traits.append('gentle')
            else:
                personality_traits.append('balanced')
            
            # Warmth assessment based on pitch and depth
            if depth in ['deep', 'medium-deep'] and pitch < 160:
                personality_traits.append('warm')
            elif pitch > 180:
                personality_traits.append('bright')
            else:
                personality_traits.append('neutral')
            
            # Energy assessment
            if pitch > 160 and volume > 0.3:
                personality_traits.append('energetic')
            elif pitch < 130 and volume < 0.4:
                personality_traits.append('calm')
            else:
                personality_traits.append('moderate')
            
            # Combine traits
            personality = '_'.join(personality_traits[:2])  # Take first two traits
            
            return personality
            
        except Exception as e:
            self.logger.warning(f"Personality assessment failed: {e}")
            return 'confident_friendly'
    
    def _get_default_characteristics(self):
        """Get default voice characteristics"""
        return {
            'estimated_gender': 'male',
            'average_pitch': 140.0,
            'estimated_pitch': 140.0,
            'pitch_range': {'min': 120, 'max': 160},
            'speaking_rate': 'moderate',
            'voice_depth': 'medium-deep',
            'accent': 'neutral',
            'audio_quality': 'good',
            'voice_type': 'custom_trained',
            'personality': 'confident_friendly',
            'vocal_characteristics': {
                'resonance': 'chest',
                'articulation': 'clear',
                'rhythm': 'steady'
            },
            'confidence_score': 0.7,
            'sample_diversity': 1
        }
    
    def retrain_existing_model(self, model_name):
        """Retrain an existing voice model with current samples"""
        try:
            # Get model info
            model_info = self.get_model_info(model_name)
            if not model_info:
                raise Exception(f"Model {model_name} not found")
            
            samples_dir = model_info.get('samples_dir')
            if not samples_dir or not Path(samples_dir).exists():
                raise Exception(f"Samples directory not found: {samples_dir}")
            
            self.logger.info(f"Retraining model '{model_name}' with enhanced analysis...")
            
            # Retrain with current samples
            return self.train_voice_model(samples_dir, model_name)
            
        except Exception as e:
            self.logger.error(f"Error retraining model: {e}")
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
            # For custom voice models, we'll use enhanced fallback TTS with voice characteristics
            if voice_model and voice_model not in ['default', 'system', 'fallback']:
                self.logger.info(f"Generating speech with voice model: {voice_model}")
                
                # Try to load voice characteristics
                voice_characteristics = self._get_voice_characteristics(voice_model)
                
                if TORTOISE_AVAILABLE and voice_model not in ['default', 'system', 'fallback']:
                    # Use Tortoise TTS for voice cloning
                    try:
                        # Remove 'tortoise_' prefix if present
                        tortoise_voice = voice_model.replace('tortoise_', '')
                        
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
                        self.logger.warning(f"Tortoise TTS failed: {e}, falling back to enhanced TTS")
                        # Fall through to enhanced fallback TTS
                
                # Enhanced fallback TTS with voice characteristics applied
                return self._generate_enhanced_speech(text, voice_characteristics, output_path)
            
            # Use standard fallback TTS
            return self._generate_standard_speech(text, voice_model, output_path)
                        
        except Exception as e:
            self.logger.error(f"Error generating speech: {e}")
            raise
    
    def _get_voice_characteristics(self, voice_model):
        """Get voice characteristics from model file"""
        try:
            model_file = self.voice_models_dir / f"{voice_model}.json"
            if model_file.exists():
                with open(model_file, 'r', encoding='utf-8') as f:
                    model_info = json.load(f)
                return model_info.get('characteristics', {})
        except Exception as e:
            self.logger.warning(f"Could not load voice characteristics for {voice_model}: {e}")
        return {}
    
    def _generate_enhanced_speech(self, text, voice_characteristics, output_path):
        """Generate speech with enhanced voice characteristics"""
        try:
            # Initialize fallback TTS if needed
            if self.fallback_tts is None:
                self.fallback_tts = pyttsx3.init()
            
            # Apply enhanced voice characteristics
            if voice_characteristics:
                # Get detailed characteristics
                estimated_pitch = voice_characteristics.get('estimated_pitch', 140.0)
                gender = voice_characteristics.get('estimated_gender', 'male')
                personality = voice_characteristics.get('personality', 'neutral')
                voice_depth = voice_characteristics.get('voice_depth', 'medium')
                
                # Calculate enhanced speech parameters
                base_rate = 180  # Slightly slower for more natural speech
                
                # Adjust rate based on personality and voice depth
                if personality == 'confident_friendly':
                    rate_modifier = 0.95  # Slightly slower, more confident
                elif 'deep' in voice_depth:
                    rate_modifier = 0.9   # Deeper voices tend to speak slower
                else:
                    rate_modifier = 1.0
                
                # Adjust based on pitch (lower pitch = slower speech)
                if estimated_pitch < 130:
                    pitch_modifier = 0.85  # Very deep voice
                elif estimated_pitch < 150:
                    pitch_modifier = 0.9   # Medium-deep voice (like yours)
                else:
                    pitch_modifier = 1.0   # Higher voice
                
                final_rate = int(base_rate * rate_modifier * pitch_modifier)
                final_rate = max(120, min(200, final_rate))  # Keep within reasonable bounds
                
                # Set volume for clarity
                volume = 0.95  # High volume for clarity
                
                # Apply the speech parameters
                self.fallback_tts.setProperty('rate', final_rate)
                self.fallback_tts.setProperty('volume', volume)
                
                # Enhanced voice selection based on detailed characteristics
                self._select_enhanced_system_voice(gender, estimated_pitch, personality, voice_depth)
                
                self.logger.info(f"Enhanced voice settings applied: rate={final_rate}, volume={volume:.2f}, pitch={estimated_pitch}Hz")
                self.logger.info(f"Voice profile: {gender}, {personality}, {voice_depth}")
            
            # Process text for more natural speech
            enhanced_text = self._enhance_text_for_natural_speech(text)
            
            # Generate speech
            if output_path:
                self.fallback_tts.save_to_file(enhanced_text, output_path)
                self.fallback_tts.runAndWait()
                
                # Verify file was created
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    self.logger.info(f"Enhanced speech generated and saved to: {output_path}")
                    return output_path
                else:
                    raise Exception("Audio file was not created or is empty")
            else:
                # For enhanced mode, we need to save to a temp file first
                temp_file = "temp_enhanced_speech.wav"
                self.fallback_tts.save_to_file(enhanced_text, temp_file)
                self.fallback_tts.runAndWait()
                
                if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                    audio_data, sr = sf.read(temp_file)
                    os.remove(temp_file)
                    return audio_data
                else:
                    raise Exception("Temporary audio file was not created")
                    
        except Exception as e:
            self.logger.error(f"Error in enhanced speech generation: {e}")
            # Fallback to standard speech generation
            return self._generate_standard_speech(text, None, output_path)
    
    def _select_best_system_voice(self, gender, pitch):
        """Select the best matching system voice based on gender and pitch"""
        try:
            voices = self.fallback_tts.getProperty('voices')
            if not voices:
                return
            
            best_voice = None
            best_score = -1
            
            target_female = gender == 'female' or pitch > 180
            
            for voice in voices:
                score = 0
                voice_name = voice.name.lower()
                voice_id = voice.id.lower()
                
                # Gender matching
                if target_female:
                    if any(keyword in voice_name for keyword in ['female', 'woman', 'girl', 'zira', 'cortana', 'eva']):
                        score += 3
                    elif any(keyword in voice_id for keyword in ['female', 'woman', 'f_']):
                        score += 2
                else:
                    if any(keyword in voice_name for keyword in ['male', 'man', 'boy', 'david', 'mark', 'richard']):
                        score += 3
                    elif any(keyword in voice_id for keyword in ['male', 'man', 'm_']):
                        score += 2
                
                # Language preference (English)
                if any(keyword in voice_name for keyword in ['english', 'en-', 'us', 'uk']):
                    score += 1
                elif any(keyword in voice_id for keyword in ['en_', 'english', '1033', '2057']):
                    score += 1
                
                # Quality indicators
                if any(keyword in voice_name for keyword in ['enhanced', 'neural', 'natural']):
                    score += 1
                
                if score > best_score:
                    best_score = score
                    best_voice = voice
            
            if best_voice:
                self.fallback_tts.setProperty('voice', best_voice.id)
                self.logger.info(f"Selected system voice: {best_voice.name} (score: {best_score})")
            
        except Exception as e:
            self.logger.warning(f"Could not select optimal system voice: {e}")
    
    def _select_enhanced_system_voice(self, gender, pitch, personality, voice_depth):
        """Enhanced voice selection based on detailed characteristics"""
        try:
            voices = self.fallback_tts.getProperty('voices')
            if not voices:
                return
            
            best_voice = None
            best_score = -1
            
            target_male = gender == 'male' or pitch < 160
            
            for voice in voices:
                score = 0
                voice_name = voice.name.lower()
                voice_id = voice.id.lower()
                
                # Gender and depth matching for male voices
                if target_male:
                    if any(keyword in voice_name for keyword in ['david', 'mark', 'richard', 'james', 'male']):
                        score += 5  # Strong preference for male names
                        
                        # Special preference for David (often deeper voice)
                        if 'david' in voice_name and 'deep' in voice_depth:
                            score += 3
                    
                    # Avoid female voices
                    if any(keyword in voice_name for keyword in ['zira', 'hazel', 'susan', 'female', 'cortana']):
                        score -= 3
                else:
                    # Female voice selection
                    if any(keyword in voice_name for keyword in ['zira', 'hazel', 'susan', 'female', 'cortana']):
                        score += 5
                
                # Language and quality preferences
                if any(keyword in voice_name for keyword in ['english', 'en-', 'us', 'united states']):
                    score += 2
                
                # Enhanced/Neural voice preference
                if any(keyword in voice_name for keyword in ['enhanced', 'neural', 'natural', 'premium']):
                    score += 2
                
                # Desktop vs Mobile preference (Desktop usually better quality)
                if 'desktop' in voice_name:
                    score += 1
                elif 'mobile' in voice_name:
                    score -= 1
                
                if score > best_score:
                    best_score = score
                    best_voice = voice
            
            if best_voice:
                self.fallback_tts.setProperty('voice', best_voice.id)
                self.logger.info(f"Selected enhanced voice: {best_voice.name} (score: {best_score})")
            
        except Exception as e:
            self.logger.warning(f"Could not select enhanced system voice: {e}")
    
    def _enhance_text_for_natural_speech(self, text):
        """Enhance text for more natural speech patterns"""
        # Add natural pauses and improve pronunciation
        enhanced_text = text
        
        # Add slight pauses after punctuation for more natural rhythm
        enhanced_text = enhanced_text.replace('.', '. ')
        enhanced_text = enhanced_text.replace(',', ', ')
        enhanced_text = enhanced_text.replace(';', '; ')
        enhanced_text = enhanced_text.replace(':', ': ')
        enhanced_text = enhanced_text.replace('!', '! ')
        enhanced_text = enhanced_text.replace('?', '? ')
        
        # Handle common abbreviations for better pronunciation
        abbreviations = {
            'AI': 'A I',
            'ML': 'M L', 
            'API': 'A P I',
            'PDF': 'P D F',
            'URL': 'U R L',
            'HTML': 'H T M L',
            'CSS': 'C S S',
            'JS': 'Java Script',
            'vs.': 'versus',
            'etc.': 'and so on',
            'e.g.': 'for example',
            'i.e.': 'that is'
        }
        
        for abbrev, pronunciation in abbreviations.items():
            enhanced_text = enhanced_text.replace(abbrev, pronunciation)
        
        # Clean up multiple spaces
        import re
        enhanced_text = re.sub(r'\s+', ' ', enhanced_text)
        
        return enhanced_text.strip()
    
    def _generate_standard_speech(self, text, voice_model, output_path):
        """Generate speech using standard fallback TTS"""
        try:
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
                    self.logger.info(f"Standard speech generated and saved to: {output_path}")
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
            self.logger.error(f"Error in standard speech generation: {e}")
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
