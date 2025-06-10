import speech_recognition as sr
import boto3
import io
import wave
import tempfile
import os
import pyaudio
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
from utils.logging import setup_logging, log_function_call, log_aws_operation

# Setup logging for this module
logger = setup_logging()

class AudioProcessor:
    @log_function_call
    def __init__(self):
        """Initialize audio processing components with proper AWS credentials and comprehensive logging"""
        
        logger.info("🎙️ Initializing AudioProcessor...")
        
        # Load environment variables
        load_dotenv()
        
        # Speech recognition setup
        logger.debug("🔧 Setting up speech recognition components...")
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        logger.info("✅ Speech recognition components initialized")
        
        # AWS credentials from environment
        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        
        logger.debug(f"AWS Region: {aws_region}")
        logger.debug(f"AWS Access Key: {aws_access_key_id[:10] if aws_access_key_id else 'NOT SET'}...")
        
        # Initialize AWS clients with explicit credentials
        try:
            if aws_access_key_id and aws_secret_access_key:
                logger.debug("🔧 Initializing AWS Polly client...")
                self.polly_client = boto3.client(
                    'polly',
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name=aws_region
                )
                
                logger.debug("🔧 Initializing AWS S3 client...")
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name=aws_region
                )
                
                self.bucket_name = os.getenv('S3_BUCKET_NAME', 'wonderstorytexttoaudiofile')
                logger.info("✅ AWS Audio clients initialized successfully")
                logger.debug(f"S3 Bucket: {self.bucket_name}")
                
            else:
                logger.warning("⚠️ AWS credentials not found - Audio features limited")
                self.polly_client = None
                self.s3_client = None
                self.bucket_name = None
                
        except Exception as e:
            logger.error(f"❌ AWS Audio initialization failed: {e}")
            logger.exception("Full exception details:")
            self.polly_client = None
            self.s3_client = None
            self.bucket_name = None
        
        logger.info("🎯 AudioProcessor initialization completed")
    
    @log_function_call
    def record_audio(self, duration=10, timeout=5):
        """Record audio from microphone with comprehensive logging"""
        logger.info(f"🎙️ Starting audio recording - Duration: {duration}s, Timeout: {timeout}s")
        
        try:
            logger.debug("🔧 Setting up microphone...")
            with self.microphone as source:
                logger.debug("🔊 Adjusting for ambient noise...")
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                logger.debug("✅ Ambient noise adjustment completed")
                
                logger.info("🔴 Recording audio...")
                # Record audio
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=duration)
                logger.info("✅ Audio recording completed")
                
                # Save audio to temporary file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                audio_file_path = f"temp_audio_{timestamp}.wav"
                
                logger.debug(f"💾 Saving audio to: {audio_file_path}")
                with open(audio_file_path, "wb") as f:
                    f.write(audio.get_wav_data())
                
                file_size = os.path.getsize(audio_file_path)
                logger.info(f"✅ Audio saved successfully - File: {audio_file_path}, Size: {file_size} bytes")
                
                return audio_file_path
                
        except sr.WaitTimeoutError:
            logger.warning("⏰ Recording timeout - No speech detected")
            st.error("❌ No speech detected. Please try again and speak clearly.")
            return None
        except Exception as e:
            logger.error(f"❌ Recording error: {str(e)}")
            logger.exception("Full exception details:")
            st.error(f"❌ Recording error: {str(e)}")
            return None
    
    @log_function_call
    def transcribe_audio(self, audio_file_path):
        """Transcribe audio to text using speech recognition with comprehensive logging"""
        logger.info(f"🔤 Starting audio transcription: {audio_file_path}")
        
        if not os.path.exists(audio_file_path):
            logger.error(f"❌ Audio file not found: {audio_file_path}")
            return {
                'success': False,
                'error': f'Audio file not found: {audio_file_path}'
            }
        
        file_size = os.path.getsize(audio_file_path)
        logger.debug(f"Audio file size: {file_size} bytes")
        
        try:
            logger.debug("📂 Loading audio file...")
            # Load audio file
            with sr.AudioFile(audio_file_path) as source:
                logger.debug("📖 Reading audio data...")
                # Read the audio data
                audio = self.recognizer.record(source)
                logger.debug("✅ Audio data loaded successfully")
                
                # Try Google Speech Recognition first (most accurate)
                try:
                    logger.debug("🌐 Attempting Google Speech Recognition...")
                    text = self.recognizer.recognize_google(audio)
                    logger.info(f"✅ Google Speech Recognition successful: '{text[:50]}...'")
                    
                    return {
                        'success': True,
                        'text': text,
                        'confidence': 0.8,
                        'method': 'google'
                    }
                    
                except sr.UnknownValueError:
                    logger.warning("⚠️ Google Speech Recognition could not understand audio")
                    
                    # Try fallback method
                    logger.debug("🔄 Trying fallback recognition method...")
                    try:
                        text = self.recognizer.recognize_sphinx(audio)
                        logger.info(f"✅ Sphinx recognition successful: '{text[:50]}...'")
                        
                        return {
                            'success': True,
                            'text': text,
                            'confidence': 0.6,
                            'method': 'sphinx'
                        }
                    except:
                        logger.error("❌ All speech recognition methods failed")
                        return {
                            'success': False,
                            'error': 'Could not understand the audio. Please speak more clearly.'
                        }
                        
                except sr.RequestError as e:
                    logger.error(f"❌ Google Speech Recognition service error: {e}")
                    
                    # Fallback to offline recognition
                    try:
                        logger.debug("🔄 Falling back to offline recognition...")
                        text = self.recognizer.recognize_sphinx(audio)
                        logger.info(f"✅ Offline recognition successful: '{text[:50]}...'")
                        
                        return {
                            'success': True,
                            'text': text,
                            'confidence': 0.6,
                            'method': 'sphinx_fallback'
                        }
                    except Exception as fallback_error:
                        logger.error(f"❌ Fallback recognition failed: {fallback_error}")
                        return {
                            'success': False,
                            'error': f'Speech recognition service error: {e}'
                        }
                
        except Exception as e:
            logger.error(f"❌ Transcription error: {str(e)}")
            logger.exception("Full exception details:")
            return {
                'success': False,
                'error': f'Transcription error: {str(e)}'
            }
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(audio_file_path):
                    os.remove(audio_file_path)
                    logger.debug(f"🗑️ Cleaned up temporary file: {audio_file_path}")
            except Exception as cleanup_error:
                logger.warning(f"⚠️ Failed to cleanup temporary file: {cleanup_error}")
    
    @log_aws_operation("Text-to-Speech")
    @log_function_call  
    def text_to_speech(self, text, voice_id='Joanna'):
        """Convert text to speech using AWS Polly with comprehensive logging"""
        
        # Temporarily disabled to avoid AWS auth errors
        logger.info("🔊 Text-to-speech temporarily disabled while fixing AWS credentials")
        st.info("🔊 Text-to-speech temporarily disabled while fixing AWS credentials")
        return None
        
        # Original TTS code (will be re-enabled once AWS auth is fully working):
        """
        logger.info(f"🔊 Starting text-to-speech conversion - Voice: {voice_id}")
        logger.debug(f"Text length: {len(text)} characters")
        logger.debug(f"Text preview: '{text[:100]}...'")
        
        if not self.polly_client:
            logger.warning("⚠️ AWS Polly not available - TTS disabled")
            st.warning("⚠️ AWS Polly not available - TTS disabled")
            return None
        
        try:
            # Limit text length for TTS
            limited_text = text[:500]
            if len(text) > 500:
                logger.debug(f"Text truncated from {len(text)} to 500 characters")
            
            logger.debug("📡 Calling AWS Polly synthesize_speech()...")
            
            # Synthesize speech
            response = self.polly_client.synthesize_speech(
                Text=limited_text,
                OutputFormat='mp3',
                VoiceId=voice_id,
                Engine='neural'
            )
            
            logger.bind(category="aws").info("✅ AWS Polly synthesis successful")
            
            # Save audio to temporary file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_file_name = f"medical_response_{timestamp}.mp3"
            
            logger.debug(f"💾 Saving TTS audio to: {audio_file_name}")
            
            with open(audio_file_name, 'wb') as file:
                audio_data = response['AudioStream'].read()
                file.write(audio_data)
            
            file_size = os.path.getsize(audio_file_name)
            logger.info(f"✅ TTS audio saved - File: {audio_file_name}, Size: {file_size} bytes")
            
            return audio_file_name
            
        except Exception as e:
            logger.error(f"❌ Text-to-speech error: {str(e)}")
            logger.exception("Full exception details:")
            logger.bind(category="aws").error(f"AWS Polly synthesis failed: {str(e)}")
            st.error(f"Text-to-speech error: {str(e)}")
            return None
        """
    
    @log_function_call
    def test_audio_system(self):
        """Test if audio system is working with comprehensive logging"""
        logger.debug("🧪 Testing audio system...")
        
        try:
            logger.debug("🎙️ Testing microphone access...")
            # Test microphone access
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            logger.info("✅ Audio system test successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Audio system test failed: {str(e)}")
            logger.exception("Full exception details:")
            return False