import speech_recognition as sr
import boto3
import io
import wave
import tempfile
import os
import pyaudio
from datetime import datetime
import streamlit as st

class AudioProcessor:
    def __init__(self):
        """Initialize audio processing components"""
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # AWS clients (reuse existing credentials)
        try:
            self.polly_client = boto3.client('polly', region_name='us-east-1')
            self.s3_client = boto3.client('s3', region_name='us-east-1')
            self.bucket_name = os.getenv('S3_BUCKET_NAME', 'wonderstorytexttoaudiofile')
        except Exception as e:
            st.error(f"AWS initialization failed: {e}")
            self.polly_client = None
            self.s3_client = None
    
    def record_audio(self, duration=10, timeout=5):
        """Record audio from microphone"""
        try:
            with self.microphone as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                st.info(f"🎙️ Recording for {duration} seconds... Speak now!")
                
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
                
                # Transcribe using Google Speech Recognition (free)
                try:
                    text = self.recognizer.recognize_google(audio)
                    return {
                        'success': True,
                        'text': text,
                        'confidence': 0.8  # Google doesn't provide confidence, so we estimate
                    }
                except sr.UnknownValueError:
                    # Try with alternative service
                    try:
                        text = self.recognizer.recognize_sphinx(audio)
                        return {
                            'success': True,
                            'text': text,
                            'confidence': 0.6  # Lower confidence for offline recognition
                        }
                    except:
                        return {
                            'success': False,
                            'error': 'Could not understand the audio. Please speak more clearly.'
                        }
                except sr.RequestError as e:
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
        """Convert text to speech using AWS Polly"""
        if not self.polly_client:
            return None
        
        try:
            # Synthesize speech
            response = self.polly_client.synthesize_speech(
                Text=text,
                OutputFormat='mp3',
                VoiceId=voice_id,
                Engine='neural'
            )
            
            # Save audio to temporary file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_file_name = f"medical_response_{timestamp}.mp3"
            
            with open(audio_file_name, 'wb') as file:
                file.write(response['AudioStream'].read())
            
            # Optionally upload to S3
            if self.s3_client and self.bucket_name:
                try:
                    self.s3_client.upload_file(
                        audio_file_name, 
                        self.bucket_name, 
                        f"medical-audio/{audio_file_name}"
                    )
                except:
                    pass  # Continue even if S3 upload fails
            
            return audio_file_name
            
        except Exception as e:
            st.error(f"Text-to-speech error: {str(e)}")
            return None
    
    def test_audio_system(self):
        """Test if audio system is working"""
        try:
            # Test microphone access
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            return True
        except Exception as e:
            return False