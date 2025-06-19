import os
import boto3
import json
import io
import base64
import tempfile
import wave
import time
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import streamlit as st
from dotenv import load_dotenv
from utils.logging import setup_logging, log_function_call, log_aws_operation

# Setup logging
logger = setup_logging()

# Try to import audio libraries with graceful fallback
try:
    import speech_recognition as sr
    import pyaudio
    AUDIO_AVAILABLE = True
    logger.info("✅ Audio libraries imported successfully")
except ImportError as e:
    AUDIO_AVAILABLE = False
    logger.warning(f"⚠️ Audio libraries not available: {str(e)}")
    sr = None
    pyaudio = None

class AudioProcessor:
    """Robust audio processor with graceful fallback when audio hardware is unavailable"""
    
    @log_function_call
    def __init__(self):
        """Initialize audio processor with robust error handling"""
        logger.info("🎙️ Initializing AudioProcessor...")
        
        # Load environment variables
        load_dotenv()
        
        # AWS credentials
        self.aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.aws_region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        self.s3_bucket = os.getenv('S3_BUCKET_NAME', 'wonderstorytexttoaudiofile')
        
        # Audio availability flags
        self.audio_libraries_available = AUDIO_AVAILABLE
        self.microphone_available = False
        self.audio_system_working = False
        
        # Initialize components
        self._initialize_aws_clients()
        self._initialize_audio_system()
        
        # Audio settings
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.audio_format = pyaudio.paInt16 if AUDIO_AVAILABLE else None
        self.channels = 1
        self.max_recording_duration = 30  # seconds
        
        # Create audio recordings directory
        self.audio_dir = "audio_recordings"
        os.makedirs(self.audio_dir, exist_ok=True)
        
        logger.info(f"🎯 AudioProcessor initialized - Audio available: {self.audio_system_working}")
    
    @log_function_call
    def _initialize_aws_clients(self):
        """Initialize AWS clients for audio processing"""
        try:
            if not self.aws_access_key_id or not self.aws_secret_access_key:
                logger.warning("⚠️ AWS credentials not found - audio transcription will be limited")
                self.polly_client = None
                self.s3_client = None
                return
            
            # Initialize AWS Polly for text-to-speech
            self.polly_client = boto3.client(
                'polly',
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.aws_region
            )
            
            # Initialize S3 client for audio file storage
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.aws_region
            )
            
            logger.info("✅ AWS clients initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize AWS clients: {str(e)}")
            self.polly_client = None
            self.s3_client = None
    
    @log_function_call
    def _initialize_audio_system(self):
        """Initialize audio system with graceful fallback"""
        if not self.audio_libraries_available:
            logger.warning("⚠️ Audio libraries not available - microphone features disabled")
            self.recognizer = None
            self.microphone = None
            return
        
        try:
            # Initialize speech recognizer
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.8
            self.recognizer.phrase_threshold = 0.3
            
            # Try to initialize microphone with error handling
            try:
                # Check if any audio input devices are available
                if not self._check_audio_devices():
                    logger.warning("⚠️ No audio input devices detected")
                    self.microphone = None
                    self.microphone_available = False
                else:
                    # Try to initialize the default microphone
                    self.microphone = sr.Microphone()
                    self.microphone_available = True
                    logger.info("✅ Microphone initialized successfully")
                    
                    # Test microphone access
                    with self.microphone as source:
                        self.recognizer.adjust_for_ambient_noise(source, duration=0.1)
                    
                    self.audio_system_working = True
                    logger.info("✅ Audio system fully operational")
                    
            except (OSError, RuntimeError) as e:
                logger.warning(f"⚠️ Microphone initialization failed: {str(e)}")
                self.microphone = None
                self.microphone_available = False
                
                # Still mark audio libraries as available for file processing
                if "No Default Input Device Available" in str(e):
                    logger.info("📝 Audio system available for file processing only")
                
        except Exception as e:
            logger.error(f"❌ Audio system initialization failed: {str(e)}")
            self.recognizer = None
            self.microphone = None
            self.microphone_available = False
            self.audio_system_working = False
    
    @log_function_call
    def _check_audio_devices(self) -> bool:
        """Check if audio input devices are available"""
        if not AUDIO_AVAILABLE:
            return False
        
        try:
            audio = pyaudio.PyAudio()
            device_count = audio.get_device_count()
            
            # Look for input devices
            input_devices = 0
            for i in range(device_count):
                device_info = audio.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    input_devices += 1
            
            audio.terminate()
            logger.debug(f"Found {input_devices} audio input devices")
            return input_devices > 0
            
        except Exception as e:
            logger.warning(f"Failed to check audio devices: {str(e)}")
            return False
    
    @log_function_call
    def test_audio_system(self) -> bool:
        """Test if audio system is working"""
        return self.audio_system_working
    
    @log_function_call
    def is_microphone_available(self) -> bool:
        """Check if microphone is available for recording"""
        return self.microphone_available and self.microphone is not None
    
    @log_function_call
    def record_audio(self, duration: int = 10) -> Optional[str]:
        """Record audio from microphone with robust error handling"""
        
        if not self.is_microphone_available():
            logger.error("❌ Microphone not available for recording")
            return None
        
        logger.info(f"🎙️ Starting audio recording for {duration} seconds...")
        
        try:
            with self.microphone as source:
                # Adjust for ambient noise
                logger.debug("🔧 Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Record audio
                logger.debug(f"🎤 Recording audio for {duration} seconds...")
                audio_data = self.recognizer.listen(
                    source, 
                    timeout=duration, 
                    phrase_time_limit=duration
                )
                
                # Save audio to file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                audio_filename = f"{self.audio_dir}/recording_{timestamp}.wav"
                
                with open(audio_filename, "wb") as f:
                    f.write(audio_data.get_wav_data())
                
                logger.info(f"✅ Audio recorded successfully: {audio_filename}")
                return audio_filename
                
        except sr.WaitTimeoutError:
            logger.warning("⚠️ Recording timeout - no speech detected")
            return None
        except Exception as e:
            logger.error(f"❌ Audio recording failed: {str(e)}")
            return None
    
    @log_aws_operation("Audio Transcription")
    @log_function_call
    def transcribe_audio(self, audio_file_path: str) -> Dict[str, Any]:
        """Transcribe audio file to text using multiple methods"""
        
        if not os.path.exists(audio_file_path):
            logger.error(f"❌ Audio file not found: {audio_file_path}")
            return {"success": False, "error": "Audio file not found"}
        
        logger.info(f"🎧 Transcribing audio file: {audio_file_path}")
        
        # Try Google Speech Recognition first (free tier)
        try:
            if self.recognizer:
                logger.debug("📡 Using Google Speech Recognition...")
                
                with sr.AudioFile(audio_file_path) as source:
                    audio_data = self.recognizer.record(source)
                
                text = self.recognizer.recognize_google(audio_data)
                
                logger.info(f"✅ Transcription successful: '{text[:50]}...'")
                return {
                    "success": True,
                    "text": text,
                    "method": "google",
                    "confidence": 0.8  # Estimated confidence
                }
                
        except sr.UnknownValueError:
            logger.warning("⚠️ Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            logger.warning(f"⚠️ Google Speech Recognition service error: {e}")
        except Exception as e:
            logger.error(f"❌ Transcription failed: {str(e)}")
        
        # Fallback: Try AWS Transcribe if available
        try:
            if self.s3_client:
                return self._transcribe_with_aws(audio_file_path)
        except Exception as e:
            logger.warning(f"⚠️ AWS transcription failed: {str(e)}")
        
        # If all methods fail
        return {
            "success": False,
            "error": "All transcription methods failed. Please check your internet connection and try again.",
            "text": "",
            "method": "none"
        }
    
    @log_aws_operation("AWS Transcribe")
    @log_function_call
    def _transcribe_with_aws(self, audio_file_path: str) -> Dict[str, Any]:
        """Transcribe using AWS Transcribe service"""
        
        if not self.s3_client:
            return {"success": False, "error": "AWS S3 client not initialized"}
        
        try:
            # Upload file to S3 first
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            s3_key = f"audio_transcription/{timestamp}_{os.path.basename(audio_file_path)}"
            
            logger.debug(f"📡 Uploading audio to S3: {s3_key}")
            self.s3_client.upload_file(audio_file_path, self.s3_bucket, s3_key)
            
            # Initialize AWS Transcribe client
            transcribe_client = boto3.client(
                'transcribe',
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.aws_region
            )
            
            # Start transcription job
            job_name = f"medical_ai_transcription_{timestamp}"
            media_uri = f"s3://{self.s3_bucket}/{s3_key}"
            
            transcribe_client.start_transcription_job(
                TranscriptionJobName=job_name,
                Media={'MediaFileUri': media_uri},
                MediaFormat='wav',
                LanguageCode='en-US',
                Settings={
                    'ShowSpeakerLabels': False,
                    'MaxSpeakerLabels': 1,
                    'ChannelIdentification': False
                }
            )
            
            # Wait for completion (simplified - in production use polling)
            logger.debug("⏳ Waiting for AWS Transcribe to complete...")
            time.sleep(10)  # Basic wait - improve this in production
            
            # Get transcription result
            response = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
            
            if response['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
                transcript_uri = response['TranscriptionJob']['Transcript']['TranscriptFileUri']
                # Fetch and parse transcript (simplified)
                return {
                    "success": True,
                    "text": "AWS transcription completed - implement full parsing",
                    "method": "aws_transcribe",
                    "confidence": 0.9
                }
            else:
                return {"success": False, "error": "AWS transcription failed or incomplete"}
                
        except Exception as e:
            logger.error(f"❌ AWS transcription error: {str(e)}")
            return {"success": False, "error": f"AWS transcription failed: {str(e)}"}
    
    @log_aws_operation("Text to Speech")
    @log_function_call
    def text_to_speech(self, text: str, voice: str = "Joanna") -> Optional[bytes]:
        """Convert text to speech using AWS Polly"""
        
        if not self.polly_client:
            logger.warning("⚠️ AWS Polly client not available")
            return None
        
        if not text.strip():
            logger.warning("⚠️ Empty text provided for TTS")
            return None
        
        logger.info(f"🎵 Converting text to speech: '{text[:50]}...'")
        
        try:
            response = self.polly_client.synthesize_speech(
                Text=text,
                OutputFormat='mp3',
                VoiceId=voice,
                Engine='neural'
            )
            
            audio_stream = response['AudioStream'].read()
            logger.info("✅ Text-to-speech conversion successful")
            return audio_stream
            
        except Exception as e:
            logger.error(f"❌ Text-to-speech conversion failed: {str(e)}")
            return None
    
    @log_function_call
    def save_audio_stream(self, audio_stream: bytes, filename: str) -> bool:
        """Save audio stream to file"""
        
        try:
            file_path = os.path.join(self.audio_dir, filename)
            
            with open(file_path, 'wb') as f:
                f.write(audio_stream)
            
            logger.info(f"✅ Audio saved: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save audio: {str(e)}")
            return False
    
    @log_function_call
    def get_audio_info(self) -> Dict[str, Any]:
        """Get comprehensive audio system information"""
        
        info = {
            "audio_libraries_available": self.audio_libraries_available,
            "microphone_available": self.microphone_available,
            "audio_system_working": self.audio_system_working,
            "aws_polly_available": self.polly_client is not None,
            "aws_s3_available": self.s3_client is not None,
            "supported_features": []
        }
        
        # Determine supported features
        if self.audio_system_working:
            info["supported_features"].append("microphone_recording")
        
        if self.audio_libraries_available:
            info["supported_features"].append("audio_file_processing")
        
        if self.polly_client:
            info["supported_features"].append("text_to_speech")
        
        if self.s3_client:
            info["supported_features"].append("cloud_audio_storage")
        
        # Add device information if available
        if AUDIO_AVAILABLE:
            try:
                audio = pyaudio.PyAudio()
                info["audio_device_count"] = audio.get_device_count()
                info["default_input_device"] = "available" if self.microphone_available else "unavailable"
                audio.terminate()
            except:
                info["audio_device_count"] = 0
                info["default_input_device"] = "unavailable"
        
        return info
    
    @log_function_call
    def cleanup_old_recordings(self, max_age_hours: int = 24):
        """Clean up old audio recording files"""
        
        if not os.path.exists(self.audio_dir):
            return
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        cleaned_count = 0
        
        try:
            for filename in os.listdir(self.audio_dir):
                file_path = os.path.join(self.audio_dir, filename)
                
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getmtime(file_path)
                    
                    if file_age > max_age_seconds:
                        os.remove(file_path)
                        cleaned_count += 1
            
            if cleaned_count > 0:
                logger.info(f"🧹 Cleaned up {cleaned_count} old audio recordings")
                
        except Exception as e:
            logger.error(f"❌ Failed to cleanup audio recordings: {str(e)}")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            # Cleanup old recordings
            self.cleanup_old_recordings()
        except:
            pass  # Ignore cleanup errors during destruction