import boto3
import json
import os
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
from utils.logging import setup_logging, log_function_call, log_aws_operation

# Setup logging for this module
logger = setup_logging()

class MedicalRAGProcessor:
    @log_function_call
    def __init__(self):
        """Initialize Medical RAG processor with explicit AWS credentials and comprehensive logging"""
        
        logger.info("🧠 Initializing MedicalRAGProcessor...")
        
        # Load environment variables
        load_dotenv()
        
        # Get AWS credentials from environment
        self.aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.aws_region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        
        logger.debug(f"AWS Region: {self.aws_region}")
        logger.debug(f"AWS Access Key: {self.aws_access_key_id[:10] if self.aws_access_key_id else 'NOT SET'}...")
        
        # Validate credentials exist
        if not self.aws_access_key_id or not self.aws_secret_access_key:
            logger.error("❌ AWS credentials not found in environment variables!")
            logger.error("Please check your .env file has AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
            self.bedrock_client = None
            self.bedrock_agent_client = None
            self.s3_client = None
            return
        
        # Initialize AWS clients with explicit credentials
        try:
            logger.debug("🔧 Initializing Bedrock Runtime client...")
            self.bedrock_client = boto3.client(
                'bedrock-runtime',
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.aws_region
            )
            logger.info("✅ Bedrock Runtime client initialized")
            
            logger.debug("🔧 Initializing Bedrock Agent Runtime client...")
            self.bedrock_agent_client = boto3.client(
                'bedrock-agent-runtime',
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.aws_region
            )
            logger.info("✅ Bedrock Agent Runtime client initialized")
            
            logger.debug("🔧 Initializing S3 client...")
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.aws_region
            )
            logger.info("✅ S3 client initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize AWS clients: {str(e)}")
            logger.exception("Full exception details:")
            self.bedrock_client = None
            self.bedrock_agent_client = None
            self.s3_client = None
        
        # Configuration
        self.knowledge_base_id = os.getenv('KNOWLEDGE_BASE_ID', 'BDHRZZXGMQ')
        self.model_id = os.getenv('BEDROCK_MODEL_ID', 'anthropic.claude-3-haiku-20240307-v1:0')
        
        logger.debug(f"Knowledge Base ID: {self.knowledge_base_id}")
        logger.debug(f"Model ID: {self.model_id}")
        
        # Emergency keywords
        self.emergency_keywords = [
            'chest pain', 'heart attack', 'stroke', 'difficulty breathing',
            'severe bleeding', 'loss of consciousness', 'overdose',
            'suicide', 'emergency', 'urgent', 'severe pain', 'can\'t breathe'
        ]
        logger.debug(f"Emergency keywords loaded: {len(self.emergency_keywords)} keywords")
        
        logger.info("🎯 MedicalRAGProcessor initialization completed")
    
    @log_function_call
    def detect_emergency_keywords(self, query):
        """Detect emergency keywords in query with logging"""
        logger.debug(f"🚨 Checking for emergency keywords in query: '{query[:50]}...'")
        
        query_lower = query.lower()
        detected = [keyword for keyword in self.emergency_keywords if keyword in query_lower]
        
        if detected:
            logger.warning(f"🚨 EMERGENCY KEYWORDS DETECTED: {detected}")
            logger.bind(category="user").warning(f"Emergency keywords in query: {detected}")
            
            return {
                'is_emergency': True,
                'keywords': detected,
                'message': '🚨 This appears to be a medical emergency. Please call 911 immediately or go to the nearest emergency room.'
            }
        
        logger.debug("✅ No emergency keywords detected")
        return {'is_emergency': False}
    
    @log_aws_operation("Knowledge Base Retrieval")
    @log_function_call
    def retrieve_medical_context(self, query, max_results=5):
        """Retrieve context from knowledge base with comprehensive logging"""
        
        logger.bind(category="aws").info(f"📚 Retrieving medical context for query: '{query[:50]}...'")
        logger.debug(f"Max results: {max_results}")
        logger.debug(f"Knowledge Base ID: {self.knowledge_base_id}")
        
        if not self.bedrock_agent_client:
            error_msg = "Bedrock Agent client not initialized. Check AWS credentials."
            logger.error(f"❌ {error_msg}")
            raise Exception(error_msg)
        
        try:
            logger.debug("📡 Calling bedrock-agent-runtime.retrieve()...")
            
            response = self.bedrock_agent_client.retrieve(
                retrievalQuery={'text': query},
                knowledgeBaseId=self.knowledge_base_id,
                retrievalConfiguration={
                    'vectorSearchConfiguration': {
                        'numberOfResults': max_results
                    }
                }
            )
            
            logger.bind(category="aws").info("✅ Knowledge base retrieval successful")
            
            retrieval_results = response.get('retrievalResults', [])
            logger.debug(f"📊 Retrieved {len(retrieval_results)} results from knowledge base")
            
            contexts = []
            
            for i, result in enumerate(retrieval_results):
                logger.debug(f"Processing retrieval result {i+1}/{len(retrieval_results)}")
                
                if 'content' in result and 'text' in result['content']:
                    context_text = result['content']['text']
                    contexts.append(context_text)
                    logger.debug(f"   Context {i+1}: {len(context_text)} characters")
                    logger.debug(f"   Preview: {context_text[:100]}...")
                else:
                    logger.warning(f"   Result {i+1}: No text content found")
            
            logger.info(f"✅ Successfully extracted {len(contexts)} contexts")
            return contexts
            
        except Exception as e:
            error_msg = f"Error retrieving medical context: {str(e)}"
            logger.error(f"❌ {error_msg}")
            logger.exception("Full exception details:")
            logger.bind(category="aws").error(f"Knowledge base retrieval failed: {str(e)}")
            raise Exception(error_msg)
    
    @log_aws_operation("Bedrock Model Invocation")
    @log_function_call
    def generate_medical_response(self, query, contexts):
        """Generate medical response using Bedrock with comprehensive logging"""
        
        logger.bind(category="aws").info(f"🤖 Generating medical response for query: '{query[:50]}...'")
        logger.debug(f"Number of contexts: {len(contexts)}")
        logger.debug(f"Model ID: {self.model_id}")
        
        if not self.bedrock_client:
            error_msg = "Bedrock client not initialized. Check AWS credentials."
            logger.error(f"❌ {error_msg}")
            raise Exception(error_msg)
        
        try:
            # Create medical prompt with context
            context_text = "\n\n".join(contexts) if contexts else "No specific context available."
            logger.debug(f"Combined context length: {len(context_text)} characters")
            
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
            
            logger.debug(f"Prompt length: {len(medical_prompt)} characters")
            
            messages = [{"role": "user", "content": [{"type": "text", "text": medical_prompt}]}]
            
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": messages,
                "temperature": 0.3,
                "top_p": 0.9
            }
            
            logger.debug("📡 Calling bedrock-runtime.invoke_model()...")
            logger.debug(f"Payload keys: {list(payload.keys())}")
            
            response = self.bedrock_client.invoke_model(
                body=json.dumps(payload),
                modelId=self.model_id,
                accept='application/json',
                contentType='application/json'
            )
            
            logger.bind(category="aws").info("✅ Bedrock model invocation successful")
            
            response_body = json.loads(response.get('body').read())
            logger.debug(f"Response body keys: {list(response_body.keys())}")
            
            medical_response = response_body.get('content')[0]['text']
            logger.info(f"✅ Generated medical response: {len(medical_response)} characters")
            logger.debug(f"Response preview: {medical_response[:200]}...")
            
            # Add medical disclaimer
            disclaimer = "\n\n⚠️ MEDICAL DISCLAIMER: This information is for educational purposes only. Always consult qualified healthcare professionals for medical advice, diagnosis, or treatment."
            
            final_response = medical_response + disclaimer
            logger.debug("✅ Medical disclaimer added to response")
            
            return final_response
            
        except Exception as e:
            error_msg = f"Error generating medical response: {str(e)}"
            logger.error(f"❌ {error_msg}")
            logger.exception("Full exception details:")
            logger.bind(category="aws").error(f"Bedrock model invocation failed: {str(e)}")
            raise Exception(error_msg)
    
    @log_function_call
    def process_medical_query(self, query, include_audio=False):
        """Process complete medical query with comprehensive logging"""
        
        logger.info(f"🔍 Processing medical query: '{query[:100]}...'")
        logger.debug(f"Include audio: {include_audio}")
        
        start_time = datetime.now()
        
        try:
            # Check for emergency keywords
            logger.debug("🚨 Checking for emergency keywords...")
            emergency_check = self.detect_emergency_keywords(query)
            
            if emergency_check['is_emergency']:
                logger.warning("🚨 Emergency detected - returning emergency response")
                
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                logger.bind(category="performance").info(f"Emergency query processing: {processing_time:.3f}s")
                
                return {
                    'success': True,
                    'response': emergency_check['message'],
                    'emergency': True,
                    'keywords': emergency_check['keywords'],
                    'confidence': 1.0,
                    'contexts': [],
                    'processing_time': processing_time
                }
            
            # Retrieve medical context
            logger.debug("📚 Retrieving medical context...")
            contexts = self.retrieve_medical_context(query)
            
            # Generate medical response
            logger.debug("🤖 Generating medical response...")
            response = self.generate_medical_response(query, contexts)
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            logger.info(f"✅ Medical query processed successfully in {processing_time:.3f}s")
            logger.bind(category="performance").info(f"Medical query processing: {processing_time:.3f}s")
            
            return {
                'success': True,
                'response': response,
                'confidence': 0.8,
                'contexts': contexts,
                'emergency': False,
                'processing_time': processing_time
            }
            
        except Exception as e:
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            logger.error(f"❌ Medical query processing failed after {processing_time:.3f}s: {str(e)}")
            logger.exception("Full exception details:")
            logger.bind(category="performance").info(f"Medical query processing (FAILED): {processing_time:.3f}s")
            
            return {
                'success': False,
                'error': str(e),
                'response': "I'm currently unable to process your medical query due to a technical issue. Please consult with a healthcare professional for medical advice.",
                'confidence': 0.0,
                'contexts': [],
                'emergency': False,
                'processing_time': processing_time
            }
    
    @log_aws_operation("AWS Connection Test")
    @log_function_call
    def test_aws_connection(self):
        """Test AWS connection for sidebar status with logging"""
        logger.debug("🧪 Testing AWS connection...")
        
        try:
            if self.s3_client:
                logger.debug("📡 Calling S3 list_buckets()...")
                buckets = self.s3_client.list_buckets()
                bucket_count = len(buckets['Buckets'])
                
                logger.info(f"✅ AWS connection test successful - Found {bucket_count} S3 buckets")
                logger.bind(category="aws").info(f"S3 connection test: {bucket_count} buckets found")
                return True
            else:
                logger.error("❌ S3 client not initialized")
                return False
                
        except Exception as e:
            logger.error(f"❌ AWS connection test failed: {e}")
            logger.bind(category="aws").error(f"S3 connection test failed: {str(e)}")
            return False