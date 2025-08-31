#!/usr/bin/env python3
"""
Audio File Converter
===================
Converts M4A files to WAV format for better compatibility with voice cloning.
"""

import os
from pathlib import Path
import logging
from typing import List, Dict

# Audio processing
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("Warning: pydub not available. Install with: pip install pydub")

class AudioConverter:
    """Convert audio files between formats"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def convert_m4a_to_wav(self, m4a_path: str, wav_path: str = None) -> str:
        """Convert M4A file to WAV format"""
        if not PYDUB_AVAILABLE:
            raise Exception("pydub is required for audio conversion. Install with: pip install pydub")
        
        m4a_file = Path(m4a_path)
        if not m4a_file.exists():
            raise FileNotFoundError(f"M4A file not found: {m4a_path}")
        
        # Generate output path if not provided
        if wav_path is None:
            wav_path = str(m4a_file.with_suffix('.wav'))
        
        try:
            self.logger.info(f"Converting {m4a_file.name} to WAV...")
            
            # Load M4A file
            audio = AudioSegment.from_file(str(m4a_file), format="m4a")
            
            # Convert to mono if stereo (better for voice cloning)
            if audio.channels > 1:
                audio = audio.set_channels(1)
                self.logger.info("Converted to mono")
            
            # Normalize sample rate to 22050 Hz (good for voice cloning)
            if audio.frame_rate != 22050:
                audio = audio.set_frame_rate(22050)
                self.logger.info(f"Resampled to 22050 Hz (was {audio.frame_rate} Hz)")
            
            # Export as WAV
            audio.export(wav_path, format="wav")
            
            # Verify the file was created
            wav_file = Path(wav_path)
            if wav_file.exists() and wav_file.stat().st_size > 0:
                duration = len(audio) / 1000.0  # seconds
                self.logger.info(f"✅ Conversion successful: {wav_file.name}")
                self.logger.info(f"Duration: {duration:.1f} seconds")
                self.logger.info(f"Sample rate: {audio.frame_rate} Hz")
                self.logger.info(f"Channels: {audio.channels}")
                return str(wav_path)
            else:
                raise Exception("WAV file was not created or is empty")
                
        except Exception as e:
            self.logger.error(f"Error converting {m4a_file.name}: {e}")
            raise
    
    def convert_directory_m4a_to_wav(self, input_dir: str, output_dir: str = None) -> List[str]:
        """Convert all M4A files in a directory to WAV format"""
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Use same directory if output not specified
        if output_dir is None:
            output_path = input_path / "converted"
        else:
            output_path = Path(output_dir)
        
        output_path.mkdir(exist_ok=True)
        
        # Find all M4A files
        m4a_files = list(input_path.glob("*.m4a"))
        if not m4a_files:
            self.logger.warning(f"No M4A files found in {input_dir}")
            return []
        
        self.logger.info(f"Found {len(m4a_files)} M4A files to convert")
        
        converted_files = []
        failed_files = []
        
        for m4a_file in m4a_files:
            try:
                wav_filename = m4a_file.stem + ".wav"
                wav_path = output_path / wav_filename
                
                converted_wav = self.convert_m4a_to_wav(str(m4a_file), str(wav_path))
                converted_files.append(converted_wav)
                
            except Exception as e:
                self.logger.error(f"Failed to convert {m4a_file.name}: {e}")
                failed_files.append(str(m4a_file))
        
        # Summary
        self.logger.info(f"\n{'='*50}")
        self.logger.info("CONVERSION SUMMARY")
        self.logger.info(f"✅ Successfully converted: {len(converted_files)} files")
        if failed_files:
            self.logger.info(f"❌ Failed conversions: {len(failed_files)} files")
            for failed in failed_files:
                self.logger.info(f"   - {Path(failed).name}")
        self.logger.info(f"📁 Output directory: {output_path}")
        self.logger.info(f"{'='*50}")
        
        return converted_files
    
    def get_audio_info(self, audio_path: str) -> Dict:
        """Get information about an audio file"""
        if not PYDUB_AVAILABLE:
            return {"error": "pydub not available"}
        
        try:
            audio_file = Path(audio_path)
            if not audio_file.exists():
                return {"error": "File not found"}
            
            # Load audio file
            audio = AudioSegment.from_file(str(audio_file))
            
            info = {
                "filename": audio_file.name,
                "format": audio_file.suffix.lower(),
                "duration_seconds": len(audio) / 1000.0,
                "duration_minutes": len(audio) / 1000.0 / 60.0,
                "sample_rate": audio.frame_rate,
                "channels": audio.channels,
                "sample_width": audio.sample_width,
                "file_size_mb": audio_file.stat().st_size / (1024 * 1024)
            }
            
            return info
            
        except Exception as e:
            return {"error": str(e)}

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert M4A files to WAV format")
    parser.add_argument("input", help="Input M4A file or directory")
    parser.add_argument("-o", "--output", help="Output WAV file or directory")
    parser.add_argument("-d", "--directory", action="store_true", help="Process entire directory")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    converter = AudioConverter()
    
    try:
        if args.directory:
            # Convert directory
            converted = converter.convert_directory_m4a_to_wav(args.input, args.output)
            print(f"\n✅ Converted {len(converted)} files successfully!")
        else:
            # Convert single file
            converted = converter.convert_m4a_to_wav(args.input, args.output)
            print(f"\n✅ Converted to: {converted}")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
