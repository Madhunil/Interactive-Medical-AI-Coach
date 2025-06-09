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

class AudioProcessor:
    def __init__(self):
        """Initialize audio processing components with proper AWS credentials"""
        
        # Load environment variables
        load_dotenv()
        
        # Speech recognition setup
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # AWS credentials from environment
        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        
        # Initialize AWS clients with explicit credentials
        try:
            if aws_access_key_id and aws_secret_access_key:
                self.polly_client = boto3.client(
                    'polly',
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name=aws_region
                )
                
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name=aws_region
                )
                
                self.bucket_name = os.getenv('S3_BUCKET_NAME', 'wonderstorytexttoaudiofile')
                print("✅ AWS Audio clients initialized successfully")
                
            else:
                print("❌ AWS credentials not found - Audio features limited")
                self.polly_client = None
                self.s3_client = None
                self.bucket_name = None
                
        except Exception as e:
            print(f"❌ AWS Audio initialization failed: {e}")
            self.polly_client = None
            self.s3_client = None
            self.bucket_name = None
    
    def record_audio(self, duration=10, timeout=5):
        """Record audio from microphone"""
        try:
            with self.microphone as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                # Record audio
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=duration)
                
                # Save audio to temporary file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                audio_file_path = f"temp_audio_{timestamp}.wav"
                
                with open(audio_file_path, "wb") as f:
                    f.write(audio.get_wav_data())
                
                return audio_file_path
                
        except sr.WaitTimeoutError:
            st.error("❌ No speech detected. Please try again and speak clearly.")
            return None
        except Exception as e:
            st.error(f"❌ Recording error: {str(e)}")
            return None
    
    def transcribe_audio(self, audio_file_path):
        """Transcribe audio to text using speech recognition"""
        try:
            # Load audio file
            with sr.AudioFile(audio_file_path) as source:
                # Read the audio data
                audio = self.recognizer.record(source)
                
                # Try Google Speech Recognition first (most accurate)
                try:
                    text = self.recognizer.recognize_google(audio)
                    return {
                        'success': True,
                        'text': text,
                        'confidence': 0.8
                    }
                except sr.UnknownValueError:
                    return {
                        'success': False,
                        'error': 'Could not understand the audio. Please speak more clearly.'
                    }
                except sr.RequestError as e:
                    # Fallback to offline recognition
                    try:
                        text = self.recognizer.recognize_sphinx(audio)
                        return {
                            'success': True,
                            'text': text,
                            'confidence': 0.6
                        }
                    except:
                        return {
                            'success': False,
                            'error': f'Speech recognition service error: {e}'
                        }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Transcription error: {str(e)}'
            }
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(audio_file_path):
                    os.remove(audio_file_path)
            except:
                pass
    
    def text_to_speech(self, text, voice_id='Joanna'):
        """Convert text to speech using AWS Polly (Disabled for now to avoid auth errors)"""
        # Temporarily disabled to avoid AWS auth errors
        st.info("🔊 Text-to-speech temporarily disabled while fixing AWS credentials")
        return None
        
        # Original TTS code (uncomment when AWS auth is fully working):
        """
        if not self.polly_client:
            st.warning("⚠️ AWS Polly not available - TTS disabled")
            return None
        
        try:
            # Synthesize speech
            response = self.polly_client.synthesize_speech(
                Text=text[:500],  # Limit text length
                OutputFormat='mp3',
                VoiceId=voice_id,
                Engine='neural'
            )
            
            # Save audio to temporary file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_file_name = f"medical_response_{timestamp}.mp3"
            
            with open(audio_file_name, 'wb') as file:
                file.write(response['AudioStream'].read())
            
            return audio_file_name
            
        except Exception as e:
            st.error(f"Text-to-speech error: {str(e)}")
            return None
        """
    
    def test_audio_system(self):
        """Test if audio system is working"""
        try:
            # Test microphone access
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            return True
        except Exception as e:
            return False