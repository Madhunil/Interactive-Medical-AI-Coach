import requests
import json
import os
from datetime import datetime
import streamlit as st
from .audio_processor import AudioProcessor

class MedicalRAGProcessor:
    def __init__(self):
        """Initialize Medical RAG processor with AWS integration"""
        
        # AWS Configuration (reuse existing WonderScribe infrastructure)
        self.aws_api_url = "https://wacnqhon34.execute-api.us-east-1.amazonaws.com/dev/"
        self.headers = {"Content-Type": "application/json"}
        
        # Initialize audio processor
        self.audio_processor = AudioProcessor()
        
        # Emergency keywords for safety
        self.emergency_keywords = [
            'chest pain', 'heart attack', 'stroke', 'difficulty breathing',
            'severe bleeding', 'loss of consciousness', 'overdose',
            'suicide', 'emergency', 'urgent', 'severe pain', 'can\'t breathe'
        ]
    
    def detect_emergency_keywords(self, query):
        """Detect emergency keywords in query"""
        query_lower = query.lower()
        detected = [keyword for keyword in self.emergency_keywords if keyword in query_lower]
        
        if detected:
            return {
                'is_emergency': True,
                'keywords': detected,
                'message': '🚨 This appears to be a medical emergency. Please call 911 immediately or go to the nearest emergency room.'
            }
        return {'is_emergency': False}
    
    def process_medical_query(self, query, include_audio=True):
        """Process medical query using existing AWS infrastructure"""
        try:
            # Check for emergency keywords first
            emergency_check = self.detect_emergency_keywords(query)
            if emergency_check['is_emergency']:
                return {
                    'success': True,
                    'response': emergency_check['message'],
                    'emergency': True,
                    'keywords': emergency_check['keywords'],
                    'confidence': 1.0,
                    'contexts': []
                }
            
            # Prepare payload for existing Lambda function
            payload = {
                "api_Path": "getMedicalResponse",  # New endpoint to add to existing Lambda
                "medical_query": query,
                "include_audio": include_audio,
                "user_id": "medical_demo_user",
                "session_id": str(datetime.now().timestamp())
            }
            
            # Call existing API Gateway endpoint
            response = requests.post(
                self.aws_api_url, 
                headers=self.headers, 
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Handle emergency response from Lambda
                if data.get('response_type') == 'emergency':
                    return {
                        'success': True,
                        'response': data.get('message', ''),
                        'emergency': True,
                        'confidence': 1.0,
                        'contexts': []
                    }
                
                # Generate audio response if requested
                audio_file = None
                if include_audio and data.get('response'):
                    audio_file = self.audio_processor.text_to_speech(
                        data.get('response', '')[:500]  # Limit audio length
                    )
                
                return {
                    'success': True,
                    'response': data.get('response', 'No response generated'),
                    'confidence': data.get('confidence', 0.7),
                    'contexts': data.get('contexts', []),
                    'audio_file': audio_file,
                    'session_id': data.get('session_id'),
                    'emergency': False
                }
            
            else:
                # Fallback response if API fails
                return self._get_fallback_response(query, include_audio)
                
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. Please try again.")
            return self._get_fallback_response(query, include_audio)
        
        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")
            return self._get_fallback_response(query, include_audio)
    
    def _get_fallback_response(self, query, include_audio=False):
        """Provide fallback response when API is unavailable"""
        fallback_response = f"""
        I understand you're asking about: "{query}"
        
        I'm currently unable to connect to the full medical knowledge base, but I can provide some general guidance:
        
        For accurate medical information, please consult with a qualified healthcare professional who can:
        • Review your specific medical history
        • Perform appropriate examinations
        • Provide personalized treatment recommendations
        
        If this is urgent, please contact your doctor or visit an emergency room.
        
        ⚠️ This system provides general information only and should not replace professional medical advice.
        """
        
        # Generate audio for fallback if requested
        audio_file = None
        if include_audio:
            audio_file = self.audio_processor.text_to_speech(fallback_response[:200])
        
        return {
            'success': True,
            'response': fallback_response,
            'confidence': 0.5,
            'contexts': [],
            'audio_file': audio_file,
            'emergency': False,
            'fallback': True
        }
    
    def test_aws_connection(self):
        """Test connection to AWS services"""
        try:
            test_payload = {
                "api_Path": "healthCheck",  # You can add this to your Lambda for testing
                "test": True
            }
            
            response = requests.post(
                self.aws_api_url,
                headers=self.headers,
                json=test_payload,
                timeout=10
            )
            
            return response.status_code == 200
        
        except Exception as e:
            return False