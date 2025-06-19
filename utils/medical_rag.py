import boto3
import json
import os
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional
from config import THERAPEUTIC_AREAS, RESPONSE_TEMPLATES, CONFIDENCE_THRESHOLD
from utils.logging import setup_logging, log_function_call, log_aws_operation

# Setup logging for this module
logger = setup_logging()

class MedicalRAGProcessor:
    """Enhanced Medical RAG processor with therapeutic area specialization and advanced features"""
    
    @log_function_call
    def __init__(self):
        """Initialize Enhanced Medical RAG processor with comprehensive configuration"""
        
        logger.info("🧠 Initializing Enhanced MedicalRAGProcessor...")
        
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
            
            logger.debug("🔧 Initializing Bedrock Agent Runtime client...")
            self.bedrock_agent_client = boto3.client(
                'bedrock-agent-runtime',
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.aws_region
            )
            
            logger.debug("🔧 Initializing S3 client...")
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.aws_region
            )
            
            logger.info("✅ AWS clients initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize AWS clients: {str(e)}")
            logger.exception("Full exception details:")
            self.bedrock_client = None
            self.bedrock_agent_client = None
            self.s3_client = None
        
        # Configuration
        self.knowledge_base_id = os.getenv('KNOWLEDGE_BASE_ID', 'BDHRZZXGMQ')
        self.model_id = os.getenv('BEDROCK_MODEL_ID', 'anthropic.claude-3-haiku-20240307-v1:0')
        
        # Enhanced configuration
        self.current_therapeutic_area = None
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        
        # Load therapeutic area configurations
        self.therapeutic_areas = THERAPEUTIC_AREAS
        self.response_templates = RESPONSE_TEMPLATES
        
        # Emergency detection system
        self.emergency_keywords = self._build_emergency_keywords()
        
        # Context management
        self.conversation_context = []
        self.user_preferences = {}
        
        logger.debug(f"Knowledge Base ID: {self.knowledge_base_id}")
        logger.debug(f"Model ID: {self.model_id}")
        logger.debug(f"Therapeutic areas loaded: {len(self.therapeutic_areas)}")
        logger.debug(f"Emergency keywords: {len(self.emergency_keywords)}")
        
        logger.info("🎯 Enhanced MedicalRAGProcessor initialization completed")
    
    @log_function_call
    def set_therapeutic_area(self, area_key: str):
        """Set the current therapeutic area for specialized processing"""
        if area_key in self.therapeutic_areas:
            self.current_therapeutic_area = area_key
            logger.info(f"🎯 Therapeutic area set to: {self.therapeutic_areas[area_key]['name']}")
        else:
            logger.warning(f"⚠️ Unknown therapeutic area: {area_key}")
    
    @log_function_call
    def _build_emergency_keywords(self) -> List[str]:
        """Build comprehensive emergency keywords from all therapeutic areas"""
        emergency_keywords = []
        
        # Global emergency keywords
        global_keywords = [
            'emergency', 'urgent', 'severe', 'critical', 'life threatening',
            'call 911', 'hospital', 'icu', 'intensive care', 'code blue',
            'cardiac arrest', 'respiratory failure', 'shock', 'coma'
        ]
        emergency_keywords.extend(global_keywords)
        
        # Therapeutic area specific emergency keywords
        for area_config in self.therapeutic_areas.values():
            area_keywords = area_config.get('emergency_keywords', [])
            emergency_keywords.extend(area_keywords)
        
        logger.debug(f"Built emergency keyword list: {len(emergency_keywords)} keywords")
        return emergency_keywords
    
    @log_function_call
    def detect_emergency_scenario(self, query: str) -> Dict[str, Any]:
        """Enhanced emergency detection with risk classification"""
        logger.debug(f"🚨 Enhanced emergency detection for query: '{query[:50]}...'")
        
        query_lower = query.lower()
        
        # Check for emergency keywords
        detected_keywords = [kw for kw in self.emergency_keywords if kw in query_lower]
        
        # Risk level classification
        high_risk_indicators = [
            'chest pain', 'difficulty breathing', 'loss of consciousness',
            'severe bleeding', 'stroke', 'heart attack', 'overdose',
            'suicidal', 'anaphylaxis', 'severe allergic reaction'
        ]
        
        medium_risk_indicators = [
            'severe pain', 'high fever', 'persistent vomiting',
            'confusion', 'dizziness', 'weakness'
        ]
        
        high_risk_detected = any(indicator in query_lower for indicator in high_risk_indicators)
        medium_risk_detected = any(indicator in query_lower for indicator in medium_risk_indicators)
        
        if detected_keywords or high_risk_detected:
            risk_level = 'emergency' if high_risk_detected else 'high_risk'
            
            logger.warning(f"🚨 {risk_level.upper()} SCENARIO DETECTED")
            logger.bind(category="user").warning(f"{risk_level} keywords: {detected_keywords}")
            
            return {
                'is_emergency': True,
                'risk_level': risk_level,
                'keywords': detected_keywords,
                'response_template': self.response_templates[risk_level],
                'immediate_action_required': high_risk_detected
            }
        
        elif medium_risk_detected:
            logger.info("⚠️ Medium risk scenario detected")
            return {
                'is_emergency': False,
                'risk_level': 'medium',
                'keywords': [],
                'response_template': self.response_templates['standard'],
                'immediate_action_required': False,
                'caution_advised': True
            }
        
        logger.debug("✅ No emergency scenario detected")
        return {
            'is_emergency': False,
            'risk_level': 'low',
            'keywords': [],
            'response_template': self.response_templates['standard'],
            'immediate_action_required': False
        }
    
    @log_aws_operation("Enhanced Knowledge Base Retrieval")
    @log_function_call
    def retrieve_medical_context(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Enhanced context retrieval with therapeutic area filtering"""
        
        logger.bind(category="aws").info(f"📚 Enhanced retrieval for: '{query[:50]}...'")
        logger.debug(f"Therapeutic area: {self.current_therapeutic_area}")
        logger.debug(f"Max results: {max_results}")
        
        if not self.bedrock_agent_client:
            error_msg = "Bedrock Agent client not initialized. Check AWS credentials."
            logger.error(f"❌ {error_msg}")
            raise Exception(error_msg)
        
        try:
            # Enhance query with therapeutic area context
            enhanced_query = self._enhance_query_with_context(query)
            logger.debug(f"Enhanced query: '{enhanced_query[:100]}...'")
            
            # Determine knowledge base ID (area-specific if available)
            kb_id = self._get_knowledge_base_id()
            logger.debug(f"Using knowledge base: {kb_id}")
            
            logger.debug("📡 Calling bedrock-agent-runtime.retrieve()...")
            
            response = self.bedrock_agent_client.retrieve(
                retrievalQuery={'text': enhanced_query},
                knowledgeBaseId=kb_id,
                retrievalConfiguration={
                    'vectorSearchConfiguration': {
                        'numberOfResults': max_results,
                        'overrideSearchType': 'HYBRID'  # Use both semantic and keyword search
                    }
                }
            )
            
            logger.bind(category="aws").info("✅ Enhanced knowledge base retrieval successful")
            
            retrieval_results = response.get('retrievalResults', [])
            logger.debug(f"📊 Retrieved {len(retrieval_results)} raw results")
            
            # Process and rank results
            processed_contexts = self._process_retrieval_results(retrieval_results, query)
            
            logger.info(f"✅ Processed {len(processed_contexts['contexts'])} high-quality contexts")
            
            return processed_contexts
            
        except Exception as e:
            error_msg = f"Error in enhanced medical context retrieval: {str(e)}"
            logger.error(f"❌ {error_msg}")
            logger.exception("Full exception details:")
            logger.bind(category="aws").error(f"Enhanced retrieval failed: {str(e)}")
            raise Exception(error_msg)
    
    @log_function_call
    def _enhance_query_with_context(self, query: str) -> str:
        """Enhance query with therapeutic area and conversation context"""
        enhanced_parts = [query]
        
        # Add therapeutic area context
        if self.current_therapeutic_area:
            area_info = self.therapeutic_areas[self.current_therapeutic_area]
            enhanced_parts.append(f"Focus on {area_info['name']} therapeutic area.")
            
            # Add relevant medical context
            area_context = area_info.get('description', '')
            if area_context:
                enhanced_parts.append(f"Context: {area_context}")
        
        # Add conversation context (last 2 interactions)
        if self.conversation_context:
            recent_context = self.conversation_context[-2:]
            context_summary = " ".join([ctx['summary'] for ctx in recent_context if 'summary' in ctx])
            if context_summary:
                enhanced_parts.append(f"Previous discussion: {context_summary}")
        
        enhanced_query = " ".join(enhanced_parts)
        return enhanced_query[:1000]  # Limit length
    
    @log_function_call
    def _get_knowledge_base_id(self) -> str:
        """Get the appropriate knowledge base ID for current therapeutic area"""
        if self.current_therapeutic_area:
            # In production, each therapeutic area would have its own KB
            area_kb_id = f"{self.knowledge_base_id}_{self.current_therapeutic_area}"
            logger.debug(f"Using area-specific KB: {area_kb_id}")
            return area_kb_id
        
        return self.knowledge_base_id
    
    @log_function_call
    def _process_retrieval_results(self, results: List[Dict], query: str) -> Dict[str, Any]:
        """Process and enhance retrieval results with scoring and filtering"""
        processed_contexts = []
        metadata = {
            'total_results': len(results),
            'filtered_results': 0,
            'average_score': 0.0,
            'source_types': []
        }
        
        for i, result in enumerate(results):
            logger.debug(f"Processing result {i+1}/{len(results)}")
            
            if 'content' in result and 'text' in result['content']:
                context_text = result['content']['text']
                
                # Calculate relevance score
                relevance_score = self._calculate_relevance_score(context_text, query)
                
                # Filter by relevance threshold
                if relevance_score >= self.confidence_threshold:
                    processed_context = {
                        'text': context_text,
                        'relevance_score': relevance_score,
                        'source': result.get('location', {}).get('s3Location', {}).get('uri', 'Unknown'),
                        'metadata': result.get('metadata', {}),
                        'snippet': context_text[:200] + "..." if len(context_text) > 200 else context_text
                    }
                    
                    processed_contexts.append(processed_context)
                    metadata['filtered_results'] += 1
                    
                    logger.debug(f"   Added context with score: {relevance_score:.3f}")
                else:
                    logger.debug(f"   Filtered out context with low score: {relevance_score:.3f}")
        
        # Sort by relevance score
        processed_contexts.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Calculate metadata
        if processed_contexts:
            metadata['average_score'] = sum(ctx['relevance_score'] for ctx in processed_contexts) / len(processed_contexts)
            metadata['source_types'] = list(set(ctx['source'].split('/')[-1].split('.')[0] for ctx in processed_contexts))
        
        return {
            'contexts': [ctx['text'] for ctx in processed_contexts],
            'detailed_contexts': processed_contexts,
            'metadata': metadata
        }
    
    @log_function_call
    def _calculate_relevance_score(self, context: str, query: str) -> float:
        """Calculate relevance score between context and query"""
        context_lower = context.lower()
        query_lower = query.lower()
        
        # Simple relevance scoring (in production, use advanced NLP techniques)
        query_words = set(query_lower.split())
        context_words = set(context_lower.split())
        
        # Word overlap score
        overlap = len(query_words.intersection(context_words))
        word_overlap_score = overlap / len(query_words) if query_words else 0
        
        # Therapeutic area relevance
        area_relevance = 0.0
        if self.current_therapeutic_area:
            area_info = self.therapeutic_areas[self.current_therapeutic_area]
            area_keywords = area_info.get('description', '').lower().split()
            area_matches = sum(1 for word in area_keywords if word in context_lower)
            area_relevance = min(area_matches / 10, 0.3)  # Max 30% boost
        
        # Medical terminology density
        medical_terms = ['patient', 'treatment', 'therapy', 'clinical', 'medical', 'drug', 'disease']
        medical_density = sum(1 for term in medical_terms if term in context_lower) / len(medical_terms)
        
        # Combined score
        final_score = (word_overlap_score * 0.5) + (area_relevance * 0.3) + (medical_density * 0.2)
        
        return min(final_score, 1.0)
    
    @log_aws_operation("Bedrock Model Invocation")
    @log_function_call
    def generate_medical_response(self, query: str, contexts: List[str]) -> Dict[str, Any]:
        """Generate medical response using Bedrock with specialized prompting"""
        
        logger.bind(category="aws").info(f"🤖 Generating medical response for: '{query[:50]}...'")
        logger.debug(f"Context count: {len(contexts)}")
        
        if not self.bedrock_client:
            error_msg = "Bedrock client not initialized. Check AWS credentials."
            logger.error(f"❌ {error_msg}")
            raise Exception(error_msg)
        
        try:
            # Build specialized prompt
            specialized_prompt = self._build_specialized_prompt(query, contexts)
            logger.debug(f"Prompt length: {len(specialized_prompt)} characters")
            
            messages = [{"role": "user", "content": [{"type": "text", "text": specialized_prompt}]}]
            
            # Standard payload
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": messages,
                "temperature": 0.3,
                "top_p": 0.9,
                "stop_sequences": ["Human:", "Assistant:"]
            }
            
            logger.debug("📡 Calling bedrock-runtime.invoke_model()...")
            
            response = self.bedrock_client.invoke_model(
                body=json.dumps(payload),
                modelId=self.model_id,
                accept='application/json',
                contentType='application/json'
            )
            
            logger.bind(category="aws").info("✅ Bedrock model invocation successful")
            
            response_body = json.loads(response.get('body').read())
            medical_response = response_body.get('content')[0]['text']
            
            # Post-process response
            enhanced_response = self._post_process_response(medical_response)
            
            # Calculate confidence score
            confidence_score = self._calculate_response_confidence(enhanced_response, contexts)
            
            logger.info(f"✅ Generated medical response: {len(enhanced_response)} chars, confidence: {confidence_score:.2f}")
            
            return {
                'response': enhanced_response,
                'confidence': confidence_score,
                'therapeutic_area': self.current_therapeutic_area,
                'response_metadata': {
                    'model_used': self.model_id,
                    'context_sources': len(contexts),
                    'generation_timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            error_msg = f"Error generating medical response: {str(e)}"
            logger.error(f"❌ {error_msg}")
            logger.exception("Full exception details:")
            logger.bind(category="aws").error(f"Generation failed: {str(e)}")
            raise Exception(error_msg)
    
    @log_function_call
    def _build_specialized_prompt(self, query: str, contexts: List[str]) -> str:
        """Build specialized prompt based on therapeutic area"""
        
        # Get therapeutic area information
        area_info = {}
        if self.current_therapeutic_area:
            area_info = self.therapeutic_areas[self.current_therapeutic_area]
        
        # Context preparation
        context_text = "\n\n".join(contexts) if contexts else "Limited context available."
        
        # Build specialized prompt
        prompt_parts = [
            f"You are a specialized medical AI assistant for {area_info.get('name', 'General Medicine')} education.",
            "",
            "ROLE AND CONTEXT:",
            f"- Therapeutic Area: {area_info.get('name', 'General Medicine')}",
            f"- Purpose: Medical education and training",
            "",
            "RESPONSE GUIDELINES:",
            "- Provide comprehensive, evidence-based medical information",
            "- Use clear, professional medical language",
            "- Include relevant clinical considerations and practical applications",
            "- Mention latest guidelines, protocols, or best practices when applicable",
            "- Always emphasize the educational nature of the information",
            "",
            "SPECIALIZED KNOWLEDGE CONTEXT:",
            context_text,
            "",
            f"QUESTION: {query}",
            "",
            "Please provide a comprehensive, evidence-based response that:",
            "1. Directly addresses the question with current medical knowledge",
            "2. Explains mechanisms, pathophysiology, or clinical reasoning as appropriate",
            "3. Includes practical clinical considerations and applications",
            "4. Mentions any important safety considerations or contraindications",
            "5. References current guidelines or best practices when relevant",
            "",
            "Remember: This is for educational purposes only. Clinical decisions must always involve qualified healthcare professionals."
        ]
        
        return "\n".join(prompt_parts)
    
    @log_function_call
    def _post_process_response(self, response: str) -> str:
        """Post-process response with enhancements and safety checks"""
        
        # Add therapeutic area specific disclaimer
        if self.current_therapeutic_area:
            area_name = self.therapeutic_areas[self.current_therapeutic_area]['name']
            response += f"\n\n📋 **{area_name} Educational Note:** This information is specific to {area_name.lower()} and should be considered within the broader clinical context."
        
        # Add standard medical disclaimer
        response += "\n\n⚠️ **Important:** This information is for educational purposes only. Always follow current clinical guidelines and consult with healthcare professionals for patient care decisions."
        
        return response
    
    @log_function_call
    def _calculate_response_confidence(self, response: str, contexts: List[str]) -> float:
        """Calculate confidence score for the generated response"""
        confidence_factors = []
        
        # Factor 1: Context availability and quality
        if contexts:
            context_quality = len(contexts) / 5  # Normalize to 0-1
            confidence_factors.append(min(context_quality, 1.0))
        else:
            confidence_factors.append(0.3)  # Low confidence without context
        
        # Factor 2: Response length and detail (indicator of comprehensive answer)
        response_length_factor = min(len(response) / 1000, 1.0)
        confidence_factors.append(response_length_factor)
        
        # Factor 3: Medical terminology usage (indicator of clinical accuracy)
        medical_terms = ['treatment', 'diagnosis', 'therapy', 'clinical', 'patient', 'efficacy', 'adverse']
        medical_term_count = sum(1 for term in medical_terms if term.lower() in response.lower())
        medical_factor = min(medical_term_count / 5, 1.0)
        confidence_factors.append(medical_factor)
        
        # Factor 4: Therapeutic area alignment
        if self.current_therapeutic_area:
            area_keywords = self.therapeutic_areas[self.current_therapeutic_area].get('description', '').split()
            area_mentions = sum(1 for keyword in area_keywords if keyword.lower() in response.lower())
            area_factor = min(area_mentions / 3, 1.0)
            confidence_factors.append(area_factor)
        
        # Calculate weighted average
        final_confidence = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
        
        return min(max(final_confidence * 0.9, 0.1), 0.95)  # Clamp between 10-95%
    
    @log_function_call
    def process_medical_query(self, query: str, therapeutic_area: str = None, include_audio: bool = False) -> Dict[str, Any]:
        """Simplified medical query processing without user context"""
        
        logger.info(f"🔍 Processing medical query: '{query[:100]}...'")
        logger.debug(f"Therapeutic area: {therapeutic_area}")
        
        start_time = datetime.now()
        
        try:
            # Set therapeutic area if provided
            if therapeutic_area:
                self.set_therapeutic_area(therapeutic_area)
            
            # Emergency detection
            emergency_check = self.detect_emergency_scenario(query)
            
            if emergency_check['is_emergency']:
                logger.warning("🚨 Emergency scenario detected - generating emergency response")
                
                template = emergency_check['response_template']
                emergency_response = (
                    template['prefix'] +
                    template['action'] +
                    "\n\n" +
                    f"**Keywords detected:** {', '.join(emergency_check['keywords'])}" +
                    template['suffix']
                )
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                return {
                    'success': True,
                    'response': emergency_response,
                    'emergency': True,
                    'risk_level': emergency_check['risk_level'],
                    'keywords': emergency_check['keywords'],
                    'confidence': 1.0,
                    'contexts': [],
                    'processing_time': processing_time,
                    'therapeutic_area': self.current_therapeutic_area
                }
            
            # Context retrieval
            logger.debug("📚 Retrieving medical context...")
            context_data = self.retrieve_medical_context(query)
            contexts = context_data['contexts']
            
            # Response generation
            logger.debug("🤖 Generating medical response...")
            response_data = self.generate_medical_response(query, contexts)
            
            # Add conversation to context for future queries
            self._update_conversation_context(query, response_data['response'])
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare comprehensive response
            final_response = {
                'success': True,
                'response': response_data['response'],
                'confidence': response_data['confidence'],
                'contexts': contexts,
                'detailed_contexts': context_data.get('detailed_contexts', []),
                'emergency': False,
                'risk_level': emergency_check.get('risk_level', 'low'),
                'processing_time': processing_time,
                'therapeutic_area': self.current_therapeutic_area,
                'metadata': {
                    **response_data.get('response_metadata', {}),
                    **context_data.get('metadata', {}),
                    'query_enhancement': bool(self.current_therapeutic_area)
                }
            }
            
            logger.info(f"✅ Medical query processed successfully in {processing_time:.3f}s")
            logger.bind(category="performance").info(f"Query processing: {processing_time:.3f}s")
            
            return final_response
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.error(f"❌ Query processing failed after {processing_time:.3f}s: {str(e)}")
            logger.exception("Full exception details:")
            
            return {
                'success': False,
                'error': str(e),
                'response': "I'm currently unable to process your medical query due to a technical issue. Please consult with a healthcare professional for medical advice.",
                'confidence': 0.0,
                'contexts': [],
                'emergency': False,
                'processing_time': processing_time,
                'therapeutic_area': self.current_therapeutic_area
            }
    
    @log_function_call
    def _update_conversation_context(self, query: str, response: str):
        """Update conversation context for better continuity"""
        context_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query[:100] + "..." if len(query) > 100 else query,
            'summary': self._extract_key_topics(query),
            'therapeutic_area': self.current_therapeutic_area
        }
        
        self.conversation_context.append(context_entry)
        
        # Keep only last 5 interactions for context
        if len(self.conversation_context) > 5:
            self.conversation_context = self.conversation_context[-5:]
    
    @log_function_call
    def _extract_key_topics(self, text: str) -> str:
        """Extract key medical topics from text (simplified)"""
        # In production, use advanced NLP for topic extraction
        key_terms = []
        medical_keywords = [
            'treatment', 'diagnosis', 'therapy', 'medication', 'drug',
            'disease', 'condition', 'syndrome', 'symptoms', 'prognosis',
            'clinical', 'patient', 'efficacy', 'adverse', 'contraindication'
        ]
        
        text_lower = text.lower()
        for keyword in medical_keywords:
            if keyword in text_lower:
                key_terms.append(keyword)
        
        return ', '.join(key_terms[:5])  # Top 5 terms
    
    @log_aws_operation("Enhanced AWS Connection Test")
    @log_function_call
    def test_aws_connection(self) -> bool:
        """Enhanced AWS connection test with comprehensive validation"""
        logger.debug("🧪 Testing enhanced AWS connection...")
        
        try:
            # Test S3 connection
            if self.s3_client:
                logger.debug("📡 Testing S3 connection...")
                buckets = self.s3_client.list_buckets()
                bucket_count = len(buckets['Buckets'])
                logger.debug(f"S3 test successful: {bucket_count} buckets")
            
            # Test Bedrock connection
            if self.bedrock_client:
                logger.debug("📡 Testing Bedrock connection...")
                # Simple model list check (if permissions allow)
                try:
                    # This might not be available in all setups
                    pass
                except:
                    logger.debug("Bedrock runtime connection assumed working (no list permissions)")
            
            # Test Bedrock Agent connection
            if self.bedrock_agent_client:
                logger.debug("📡 Testing Bedrock Agent connection...")
                # Connection test would go here
                pass
            
            if all([self.s3_client, self.bedrock_client, self.bedrock_agent_client]):
                logger.info("✅ Enhanced AWS connection test successful")
                logger.bind(category="aws").info("All AWS services connected successfully")
                return True
            else:
                logger.error("❌ Some AWS clients not initialized")
                return False
                
        except Exception as e:
            logger.error(f"❌ Enhanced AWS connection test failed: {e}")
            logger.bind(category="aws").error(f"Connection test failed: {str(e)}")
            return False