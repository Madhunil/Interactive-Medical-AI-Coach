import boto3
import json
import os
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

class MedicalRAGProcessor:
    def __init__(self):
        """Initialize Medical RAG processor with explicit AWS credentials"""
        
        # Load environment variables
        load_dotenv()
        
        # Get AWS credentials from environment
        self.aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.aws_region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        
        # Validate credentials exist
        if not self.aws_access_key_id or not self.aws_secret_access_key:
            st.error("❌ AWS credentials not found in .env file!")
            st.error("Please check your .env file has AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
            self.bedrock_client = None
            self.bedrock_agent_client = None
            self.s3_client = None
            return
        
        # Initialize AWS clients with explicit credentials
        try:
            self.bedrock_client = boto3.client(
                'bedrock-runtime',
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.aws_region
            )
            
            self.bedrock_agent_client = boto3.client(
                'bedrock-agent-runtime',
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.aws_region
            )
            
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.aws_region
            )
            
            print("✅ AWS RAG clients initialized successfully")
            
        except Exception as e:
            st.error(f"❌ Failed to initialize AWS clients: {str(e)}")
            self.bedrock_client = None
            self.bedrock_agent_client = None
            self.s3_client = None
        
        # Configuration
        self.knowledge_base_id = os.getenv('KNOWLEDGE_BASE_ID', 'BDHRZZXGMQ')
        self.model_id = os.getenv('BEDROCK_MODEL_ID', 'anthropic.claude-3-haiku-20240307-v1:0')
        
        # Emergency keywords
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
    
    def retrieve_medical_context(self, query, max_results=5):
        """Retrieve context from knowledge base"""
        
        if not self.bedrock_agent_client:
            raise Exception("Bedrock Agent client not initialized. Check AWS credentials.")
        
        try:
            response = self.bedrock_agent_client.retrieve(
                retrievalQuery={'text': query},
                knowledgeBaseId=self.knowledge_base_id,
                retrievalConfiguration={
                    'vectorSearchConfiguration': {
                        'numberOfResults': max_results
                    }
                }
            )
            
            retrieval_results = response.get('retrievalResults', [])
            contexts = []
            
            for result in retrieval_results:
                if 'content' in result and 'text' in result['content']:
                    contexts.append(result['content']['text'])
            
            return contexts
            
        except Exception as e:
            error_msg = f"Error retrieving medical context: {str(e)}"
            st.error(error_msg)
            raise Exception(error_msg)
    
    def generate_medical_response(self, query, contexts):
        """Generate medical response using Bedrock"""
        
        if not self.bedrock_client:
            raise Exception("Bedrock client not initialized. Check AWS credentials.")
        
        try:
            # Create medical prompt with context
            context_text = "\n\n".join(contexts) if contexts else "No specific context available."
            
            medical_prompt = f"""
            You are a medical AI assistant providing evidence-based health information.
            
            IMPORTANT GUIDELINES:
            - Provide general health information only
            - Always recommend consulting healthcare professionals
            - Never provide definitive diagnoses
            - Use cautious, informative language
            - Include appropriate medical disclaimers
            
            Medical Knowledge Context:
            {context_text}
            
            Patient Question: {query}
            
            Please provide helpful, evidence-based information while emphasizing the need for professional medical consultation.
            """
            
            messages = [{"role": "user", "content": [{"type": "text", "text": medical_prompt}]}]
            
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": messages,
                "temperature": 0.3,
                "top_p": 0.9
            }
            
            response = self.bedrock_client.invoke_model(
                body=json.dumps(payload),
                modelId=self.model_id,
                accept='application/json',
                contentType='application/json'
            )
            
            response_body = json.loads(response.get('body').read())
            medical_response = response_body.get('content')[0]['text']
            
            # Add medical disclaimer
            disclaimer = "\n\n⚠️ MEDICAL DISCLAIMER: This information is for educational purposes only. Always consult qualified healthcare professionals for medical advice, diagnosis, or treatment."
            
            final_response = medical_response + disclaimer
            
            return final_response
            
        except Exception as e:
            error_msg = f"Error generating medical response: {str(e)}"
            st.error(error_msg)
            raise Exception(error_msg)
    
    def process_medical_query(self, query, include_audio=False):
        """Process complete medical query"""
        
        try:
            # Check for emergency keywords
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
            
            # Retrieve medical context
            contexts = self.retrieve_medical_context(query)
            
            # Generate medical response
            response = self.generate_medical_response(query, contexts)
            
            return {
                'success': True,
                'response': response,
                'confidence': 0.8,
                'contexts': contexts,
                'emergency': False
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response': "I'm currently unable to process your medical query due to a technical issue. Please consult with a healthcare professional for medical advice.",
                'confidence': 0.0,
                'contexts': [],
                'emergency': False
            }
    
    def test_aws_connection(self):
        """Test AWS connection for sidebar status"""
        try:
            if self.s3_client:
                self.s3_client.list_buckets()
                return True
            return False
        except Exception as e:
            print(f"AWS connection test failed: {e}")
            return False