import boto3
import json
import os
import tempfile
import requests
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import THERAPEUTIC_AREAS, RESPONSE_TEMPLATES, CONFIDENCE_THRESHOLD
from utils.logging import setup_logging, log_function_call, log_aws_operation

# Try to import Lambda monitoring (optional)
try:
    from utils.lambda_monitor import lambda_monitor, monitor_lambda_call
    LAMBDA_MONITORING_AVAILABLE = True
except ImportError:
    LAMBDA_MONITORING_AVAILABLE = False
    # Create dummy decorator if monitoring not available
    def monitor_lambda_call(function_name: str):
        def decorator(func):
            return func
        return decorator

# Try to import PDF processing libraries with graceful fallback
try:
    import PyPDF2
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Try to import sentence transformers for better embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Setup logging for this module
logger = setup_logging()

class MedicalRAGProcessor:
    """Enhanced Medical RAG processor with S3 document processing and Lambda integration"""
    
    @log_function_call
    def __init__(self):
        """Initialize Enhanced Medical RAG processor with S3 document processing"""
        
        logger.info("🧠 Initializing Enhanced MedicalRAGProcessor with S3 document processing...")
        
        # Load environment variables
        load_dotenv()
        
        # Get AWS credentials from environment
        self.aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.aws_region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        
        logger.debug(f"AWS Region: {self.aws_region}")
        logger.debug(f"AWS Access Key: {self.aws_access_key_id[:10] if self.aws_access_key_id else 'NOT SET'}...")
        
        # S3 Configuration for documents
        self.s3_bucket = 'jnjtraining'  # Based on the S3 path shown in the image
        self.s3_document_key = 'etlp/J&J_2025_Workbook.pdf'  # Document path from the image
        
        # Lambda configuration
        self.lambda_function_name = os.getenv('LAMBDA_FUNCTION_NAME', 'vector_db_connector_lambda')
        self.lambda_api_gateway_url = os.getenv('LAMBDA_API_GATEWAY_URL')
        
        # Validate credentials exist
        if not self.aws_access_key_id or not self.aws_secret_access_key:
            logger.error("❌ AWS credentials not found in environment variables!")
            self.bedrock_client = None
            self.s3_client = None
            self.lambda_client = None
            return
        
        # Initialize AWS clients
        self._initialize_aws_clients()
        
        # Configuration
        self.model_id = os.getenv('BEDROCK_MODEL_ID', 'anthropic.claude-3-haiku-20240307-v1:0')
        
        # Enhanced configuration
        self.current_therapeutic_area = None
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        
        # Load therapeutic area configurations
        self.therapeutic_areas = THERAPEUTIC_AREAS
        self.response_templates = RESPONSE_TEMPLATES
        
        # Emergency detection system
        self.emergency_keywords = self._build_emergency_keywords()
        
        # Document processing
        self.document_chunks = []
        self.document_embeddings = None
        self.vectorizer = None
        self.semantic_model = None
        
        # Initialize document processing
        self._initialize_document_processing()
        
        # Context management
        self.conversation_context = []
        self.user_preferences = {}
        
        logger.debug(f"S3 Bucket: {self.s3_bucket}")
        logger.debug(f"S3 Document: {self.s3_document_key}")
        logger.debug(f"Model ID: {self.model_id}")
        logger.debug(f"Therapeutic areas loaded: {len(self.therapeutic_areas)}")
        logger.debug(f"Emergency keywords: {len(self.emergency_keywords)}")
        
        logger.info("🎯 Enhanced MedicalRAGProcessor initialization completed")
    
    @log_function_call
    def _initialize_aws_clients(self):
        """Initialize AWS clients with explicit credentials"""
        try:
            logger.debug("🔧 Initializing AWS clients...")
            
            # Initialize Bedrock Runtime client
            self.bedrock_client = boto3.client(
                'bedrock-runtime',
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.aws_region
            )
            
            # Initialize S3 client for document access
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.aws_region
            )
            
            # Initialize Lambda client for API calls
            self.lambda_client = boto3.client(
                'lambda',
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.aws_region
            )
            
            logger.info("✅ AWS clients initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize AWS clients: {str(e)}")
            logger.exception("Full exception details:")
            self.bedrock_client = None
            self.s3_client = None
            self.lambda_client = None
    
    @log_function_call
    def _initialize_document_processing(self):
        """Initialize document processing components"""
        try:
            logger.info("📚 Initializing document processing...")
            
            # Initialize TF-IDF vectorizer for basic similarity
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Try to initialize semantic model if available
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    logger.debug("🤖 Loading sentence transformer model...")
                    self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                    logger.info("✅ Semantic model loaded successfully")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load semantic model: {e}")
                    self.semantic_model = None
            else:
                logger.warning("⚠️ Sentence transformers not available, using TF-IDF only")
            
            # Load and process the document
            self._load_and_process_document()
            
        except Exception as e:
            logger.error(f"❌ Document processing initialization failed: {e}")
            logger.exception("Full exception details:")
    
    @log_aws_operation("S3 Document Download")
    @log_function_call
    def _load_and_process_document(self):
        """Load and process the medical document from S3"""
        try:
            logger.info(f"📥 Downloading document from S3: s3://{self.s3_bucket}/{self.s3_document_key}")
            
            if not self.s3_client:
                raise Exception("S3 client not initialized")
            
            # Download document to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                self.s3_client.download_file(
                    Bucket=self.s3_bucket,
                    Key=self.s3_document_key,
                    Filename=temp_file.name
                )
                temp_file_path = temp_file.name
            
            logger.info(f"✅ Document downloaded to: {temp_file_path}")
            
            # Process the PDF document
            document_text = self._extract_text_from_pdf(temp_file_path)
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            if document_text:
                logger.info(f"📖 Extracted text: {len(document_text)} characters")
                
                # Chunk the document
                self.document_chunks = self._chunk_document(document_text)
                logger.info(f"📄 Created {len(self.document_chunks)} chunks")
                
                # Create embeddings
                self._create_embeddings()
                
                logger.info("✅ Document processing completed successfully")
            else:
                logger.error("❌ No text extracted from document")
                
        except Exception as e:
            logger.error(f"❌ Failed to load and process document: {e}")
            logger.exception("Full exception details:")
            # Create fallback content
            self._create_fallback_content()
    
    @log_function_call
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using multiple methods"""
        logger.debug(f"📖 Extracting text from PDF: {pdf_path}")
        
        if not PDF_AVAILABLE:
            logger.error("❌ PDF processing libraries not available")
            return ""
        
        text_content = ""
        
        # Try pdfplumber first (better for complex layouts)
        try:
            import pdfplumber
            logger.debug("🔧 Using pdfplumber for text extraction...")
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            
            if text_content.strip():
                logger.info("✅ Text extracted successfully with pdfplumber")
                return text_content
                
        except Exception as e:
            logger.warning(f"⚠️ pdfplumber extraction failed: {e}")
        
        # Fallback to PyPDF2
        try:
            logger.debug("🔧 Fallback to PyPDF2...")
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            
            if text_content.strip():
                logger.info("✅ Text extracted successfully with PyPDF2")
                return text_content
                
        except Exception as e:
            logger.error(f"❌ PyPDF2 extraction failed: {e}")
        
        return text_content
    
    @log_function_call
    def _chunk_document(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """Split document into chunks with metadata"""
        logger.debug(f"📄 Chunking document: {len(text)} chars, chunk_size={chunk_size}, overlap={overlap}")
        
        chunks = []
        words = text.split()
        
        # Estimate words per chunk
        words_per_chunk = chunk_size // 5  # Rough estimate: 5 chars per word
        overlap_words = overlap // 5
        
        for i in range(0, len(words), words_per_chunk - overlap_words):
            chunk_words = words[i:i + words_per_chunk]
            chunk_text = ' '.join(chunk_words)
            
            if chunk_text.strip():
                chunk_info = {
                    'text': chunk_text,
                    'chunk_id': len(chunks),
                    'start_word': i,
                    'end_word': i + len(chunk_words),
                    'word_count': len(chunk_words),
                    'char_count': len(chunk_text)
                }
                chunks.append(chunk_info)
        
        logger.debug(f"✅ Created {len(chunks)} chunks")
        return chunks
    
    @log_function_call
    def _create_embeddings(self):
        """Create embeddings for document chunks"""
        logger.debug("🔮 Creating embeddings for document chunks...")
        
        if not self.document_chunks:
            logger.warning("⚠️ No document chunks available for embedding")
            return
        
        chunk_texts = [chunk['text'] for chunk in self.document_chunks]
        
        try:
            # Create TF-IDF embeddings (always available)
            logger.debug("📊 Creating TF-IDF embeddings...")
            self.tfidf_matrix = self.vectorizer.fit_transform(chunk_texts)
            logger.info(f"✅ TF-IDF embeddings created: {self.tfidf_matrix.shape}")
            
            # Create semantic embeddings if available
            if self.semantic_model:
                logger.debug("🧠 Creating semantic embeddings...")
                self.semantic_embeddings = self.semantic_model.encode(chunk_texts)
                logger.info(f"✅ Semantic embeddings created: {self.semantic_embeddings.shape}")
            else:
                self.semantic_embeddings = None
                
        except Exception as e:
            logger.error(f"❌ Failed to create embeddings: {e}")
            self.tfidf_matrix = None
            self.semantic_embeddings = None
    
    @log_function_call
    def _create_fallback_content(self):
        """Create fallback content when document processing fails"""
        logger.info("🆘 Creating fallback medical content...")
        
        fallback_content = [
            {
                'text': """
                Medical AI Coach Educational Content - General Medicine
                
                This system provides educational information about medical topics including:
                - Disease diagnosis and treatment approaches
                - Pharmacology and drug mechanisms
                - Clinical guidelines and best practices
                - Therapeutic areas and specializations
                
                Key Medical Concepts:
                - Evidence-based medicine and clinical decision making
                - Patient safety and quality improvement
                - Medical ethics and professional standards
                - Continuing medical education and lifelong learning
                """,
                'chunk_id': 0,
                'start_word': 0,
                'end_word': 100,
                'word_count': 100,
                'char_count': 500
            },
            {
                'text': """
                Therapeutic Areas Covered:
                
                Oncology: Cancer treatment, immunotherapy, targeted therapies
                Neuroscience: CNS disorders, neurodegenerative diseases
                Cardiovascular: Heart disease, hypertension, lipid disorders
                Immunology: Autoimmune diseases, inflammatory conditions
                Pulmonary: Respiratory disorders, breathing conditions
                Infectious Diseases: Antimicrobial therapy, infection control
                Emergency Medicine: Critical care, trauma management
                Pharmacology: Drug mechanisms, interactions, therapeutic monitoring
                """,
                'chunk_id': 1,
                'start_word': 100,
                'end_word': 200,
                'word_count': 100,
                'char_count': 500
            }
        ]
        
        self.document_chunks = fallback_content
        
        # Create basic embeddings for fallback content
        try:
            chunk_texts = [chunk['text'] for chunk in self.document_chunks]
            self.tfidf_matrix = self.vectorizer.fit_transform(chunk_texts)
            logger.info("✅ Fallback content and embeddings created")
        except Exception as e:
            logger.error(f"❌ Failed to create fallback embeddings: {e}")
    
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
    
    @log_function_call
    def retrieve_medical_context(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Retrieve relevant context from processed documents using similarity search"""
        
        logger.info(f"📚 Retrieving context for: '{query[:50]}...'")
        logger.debug(f"Therapeutic area: {self.current_therapeutic_area}")
        logger.debug(f"Max results: {max_results}")
        
        if not self.document_chunks:
            logger.warning("⚠️ No document chunks available")
            return {
                'contexts': [],
                'detailed_contexts': [],
                'metadata': {'total_results': 0, 'method': 'none'}
            }
        
        try:
            # Enhance query with therapeutic area context
            enhanced_query = self._enhance_query_with_context(query)
            logger.debug(f"Enhanced query: '{enhanced_query[:100]}...'")
            
            # Perform similarity search
            similar_chunks = self._find_similar_chunks(enhanced_query, max_results)
            
            # Process results
            processed_contexts = self._process_similarity_results(similar_chunks, query)
            
            logger.info(f"✅ Retrieved {len(processed_contexts['contexts'])} relevant contexts")
            
            return processed_contexts
            
        except Exception as e:
            error_msg = f"Error in context retrieval: {str(e)}"
            logger.error(f"❌ {error_msg}")
            logger.exception("Full exception details:")
            
            return {
                'contexts': [],
                'detailed_contexts': [],
                'metadata': {'total_results': 0, 'method': 'error', 'error': str(e)}
            }
    
    @log_function_call
    def _find_similar_chunks(self, query: str, max_results: int) -> List[Tuple[int, float]]:
        """Find similar chunks using available similarity methods"""
        logger.debug(f"🔍 Finding similar chunks for query: '{query[:50]}...'")
        
        similarities = []
        
        try:
            # Use semantic similarity if available
            if self.semantic_model and self.semantic_embeddings is not None:
                logger.debug("🧠 Using semantic similarity search...")
                
                query_embedding = self.semantic_model.encode([query])
                semantic_similarities = cosine_similarity(query_embedding, self.semantic_embeddings)[0]
                
                # Get top results
                top_indices = np.argsort(semantic_similarities)[::-1][:max_results * 2]
                
                for idx in top_indices:
                    if semantic_similarities[idx] > 0.1:  # Minimum threshold
                        similarities.append((idx, semantic_similarities[idx]))
                
                logger.debug(f"Semantic similarity found {len(similarities)} results")
            
            # Use TF-IDF similarity as fallback or additional method
            if self.tfidf_matrix is not None and len(similarities) < max_results:
                logger.debug("📊 Using TF-IDF similarity search...")
                
                query_vector = self.vectorizer.transform([query])
                tfidf_similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
                
                # Get top results
                top_indices = np.argsort(tfidf_similarities)[::-1][:max_results * 2]
                
                for idx in top_indices:
                    if tfidf_similarities[idx] > 0.05:  # Lower threshold for TF-IDF
                        # Avoid duplicates from semantic search
                        if not any(sim_idx == idx for sim_idx, _ in similarities):
                            similarities.append((idx, tfidf_similarities[idx]))
                
                logger.debug(f"TF-IDF similarity found {len(similarities)} total results")
            
            # Sort by similarity score and limit results
            similarities.sort(key=lambda x: x[1], reverse=True)
            similarities = similarities[:max_results]
            
            logger.debug(f"✅ Final similarity results: {len(similarities)}")
            return similarities
            
        except Exception as e:
            logger.error(f"❌ Similarity search failed: {e}")
            return []
    
    @log_function_call
    def _process_similarity_results(self, similar_chunks: List[Tuple[int, float]], query: str) -> Dict[str, Any]:
        """Process similarity search results"""
        
        contexts = []
        detailed_contexts = []
        
        for chunk_idx, similarity_score in similar_chunks:
            if chunk_idx < len(self.document_chunks):
                chunk = self.document_chunks[chunk_idx]
                
                # Apply relevance threshold
                if similarity_score >= self.confidence_threshold * 0.5:  # Lower threshold for document search
                    
                    context_text = chunk['text']
                    
                    # Add to results
                    contexts.append(context_text)
                    
                    detailed_context = {
                        'text': context_text,
                        'similarity_score': similarity_score,
                        'chunk_id': chunk['chunk_id'],
                        'source': f"Document chunk {chunk['chunk_id']}",
                        'metadata': {
                            'word_count': chunk['word_count'],
                            'char_count': chunk['char_count']
                        },
                        'snippet': context_text[:200] + "..." if len(context_text) > 200 else context_text
                    }
                    
                    detailed_contexts.append(detailed_context)
                    
                    logger.debug(f"   Added chunk {chunk_idx} with score: {similarity_score:.3f}")
        
        # Calculate metadata
        metadata = {
            'total_results': len(contexts),
            'average_score': np.mean([dc['similarity_score'] for dc in detailed_contexts]) if detailed_contexts else 0.0,
            'method': 'semantic+tfidf' if self.semantic_model else 'tfidf',
            'source_type': 'processed_document'
        }
        
        return {
            'contexts': contexts,
            'detailed_contexts': detailed_contexts,
            'metadata': metadata
        }
    
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
        context_text = "\n\n".join(contexts) if contexts else "Limited context available from processed medical documents."
        
        # Build specialized prompt
        prompt_parts = [
            f"You are a specialized medical AI assistant for {area_info.get('name', 'General Medicine')} education.",
            "",
            "ROLE AND CONTEXT:",
            f"- Therapeutic Area: {area_info.get('name', 'General Medicine')}",
            f"- Purpose: Medical education and training",
            f"- Document Source: J&J 2025 Medical Workbook",
            "",
            "RESPONSE GUIDELINES:",
            "- Provide comprehensive, evidence-based medical information",
            "- Use clear, professional medical language",
            "- Include relevant clinical considerations and practical applications",
            "- Mention latest guidelines, protocols, or best practices when applicable",
            "- Always emphasize the educational nature of the information",
            "- Reference the source document when relevant",
            "",
            "DOCUMENT KNOWLEDGE CONTEXT:",
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
            "6. Cites information from the J&J workbook when applicable",
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
        
        # Add document source reference
        response += f"\n\n📚 **Source:** Based on J&J 2025 Medical Workbook and current medical knowledge."
        
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
    
    @monitor_lambda_call("wonderscribeconnectVDB")
    @log_aws_operation("Lambda Function Integration")
    @log_function_call
    def call_lambda_function(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Call the wonderscribeconnectVDB Lambda function for additional processing"""
        
        logger.info("🔧 Calling wonderscribeconnectVDB Lambda function...")
        
        if not self.lambda_client:
            logger.warning("⚠️ Lambda client not available")
            return {'success': False, 'error': 'Lambda client not initialized'}
        
        # Use the specific Lambda function name from AWS console
        lambda_function_name = 'wonderscribeconnectVDB'
        api_gateway_url = 'https://wacnjhqh34.execute-api.us-east-1.amazonaws.com/dev/'
        
        start_time = datetime.now()
        
        try:
            # Call via API Gateway (preferred method based on console setup)
            logger.debug(f"📡 Calling Lambda via API Gateway: {api_gateway_url}")
            
            response = requests.post(
                api_gateway_url,
                json={'body': json.dumps(payload)},
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': 'MedicalAICoach/1.0'
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                processing_time = (datetime.now() - start_time).total_seconds()
                
                logger.info(f"✅ Lambda API Gateway call successful ({processing_time:.2f}s)")
                
                # Record successful invocation
                if LAMBDA_MONITORING_AVAILABLE:
                    lambda_monitor.record_invocation(
                        lambda_function_name, processing_time, True
                    )
                
                return {'success': True, 'data': result, 'processing_time': processing_time}
            else:
                processing_time = (datetime.now() - start_time).total_seconds()
                error_msg = f'API Gateway error: {response.status_code}'
                
                logger.error(f"❌ Lambda API Gateway call failed: {response.status_code}")
                logger.debug(f"Response content: {response.text}")
                
                # Record failed invocation
                if LAMBDA_MONITORING_AVAILABLE:
                    lambda_monitor.record_invocation(
                        lambda_function_name, processing_time, False, error_msg
                    )
                
                return {'success': False, 'error': error_msg, 'processing_time': processing_time}
                
        except requests.exceptions.RequestException as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.warning(f"⚠️ API Gateway call failed, trying direct Lambda: {e}")
            
            # Fallback to direct Lambda invocation
            try:
                logger.debug(f"📡 Calling Lambda function directly: {lambda_function_name}")
                
                fallback_start = datetime.now()
                response = self.lambda_client.invoke(
                    FunctionName=lambda_function_name,
                    InvocationType='RequestResponse',
                    Payload=json.dumps({'body': json.dumps(payload)})
                )
                
                response_payload = json.loads(response['Payload'].read().decode('utf-8'))
                fallback_time = (datetime.now() - fallback_start).total_seconds()
                total_time = (datetime.now() - start_time).total_seconds()
                
                if response['StatusCode'] == 200:
                    logger.info(f"✅ Lambda direct call successful ({fallback_time:.2f}s)")
                    
                    # Record successful fallback invocation
                    if LAMBDA_MONITORING_AVAILABLE:
                        lambda_monitor.record_invocation(
                            f"{lambda_function_name}_direct", total_time, True
                        )
                    
                    return {'success': True, 'data': response_payload, 'processing_time': total_time, 'method': 'direct'}
                else:
                    error_msg = f'Lambda error: {response["StatusCode"]}'
                    logger.error(f"❌ Lambda direct call failed: {response['StatusCode']}")
                    
                    # Record failed fallback invocation
                    if LAMBDA_MONITORING_AVAILABLE:
                        lambda_monitor.record_invocation(
                            f"{lambda_function_name}_direct", total_time, False, error_msg
                        )
                    
                    return {'success': False, 'error': error_msg, 'processing_time': total_time}
                    
            except Exception as direct_error:
                total_time = (datetime.now() - start_time).total_seconds()
                error_msg = f"Direct Lambda call also failed: {direct_error}"
                logger.error(f"❌ {error_msg}")
                
                # Record complete failure
                if LAMBDA_MONITORING_AVAILABLE:
                    lambda_monitor.record_invocation(
                        lambda_function_name, total_time, False, error_msg
                    )
                
                return {'success': False, 'error': error_msg, 'processing_time': total_time}
        
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Lambda function call failed: {e}"
            logger.error(f"❌ {error_msg}")
            
            # Record general failure
            if LAMBDA_MONITORING_AVAILABLE:
                lambda_monitor.record_invocation(
                    lambda_function_name, processing_time, False, error_msg
                )
            
            return {'success': False, 'error': error_msg, 'processing_time': processing_time}
    
    @log_aws_operation("Lambda Medical Query Processing")
    @log_function_call
    def call_lambda_for_medical_query(self, query: str, therapeutic_area: str = None) -> Dict[str, Any]:
        """Call Lambda function specifically for medical query processing using existing getStory endpoint"""
        
        logger.info(f"🏥 Calling Lambda for medical query: '{query[:50]}...'")
        
        # Prepare payload in the format expected by the Lambda function's getStory method
        lambda_payload = {
            'api_Path': 'getStory',  # Use existing story generation endpoint
            'character_type': 'Medical Professional',
            'age': '30',
            'height': 'average',
            'hair_color': 'brown',
            'eye_color': 'brown',
            'story_type': 'medical education',
            'main_character': 'Healthcare Professional',
            'story_theme': query,  # Use the medical query as the story theme
            'moral_lesson': 'evidence-based medical practice',
            'setting': 'clinical environment',
            'word_count': '500',
            'story_lang': 'English',
            'therapeutic_area': therapeutic_area or 'general_medicine'
        }
        
        try:
            result = self.call_lambda_function(lambda_payload)
            
            if result['success']:
                lambda_response = result['data']
                
                # Parse the Lambda response (it returns story data)
                if 'body' in lambda_response:
                    body_data = json.loads(lambda_response['body'])
                    
                    # Extract story texts which contain medical information
                    story_texts = body_data.get('story_texts', [])
                    captions = body_data.get('captions', [])
                    
                    # Combine story texts into medical response
                    if story_texts:
                        medical_response = "\n\n".join(story_texts)
                        
                        return {
                            'success': True,
                            'response': medical_response,
                            'captions': captions,
                            'source': 'lambda_function',
                            'method': 'getStory_medical_adaptation'
                        }
                
                return {
                    'success': False,
                    'error': 'Invalid response format from Lambda function'
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"❌ Lambda medical query processing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    @log_aws_operation("Lambda Enhanced Processing")
    @log_function_call
    def get_enhanced_response_with_lambda(self, query: str, local_contexts: List[str], therapeutic_area: str = None) -> Dict[str, Any]:
        """Get enhanced response by combining local processing with Lambda function"""
        
        logger.info("🔄 Getting enhanced response with Lambda integration...")
        
        try:
            # First, try to get response from Lambda
            lambda_result = self.call_lambda_for_medical_query(query, therapeutic_area)
            
            if lambda_result['success']:
                lambda_response = lambda_result['response']
                
                # Combine Lambda response with local context
                if local_contexts:
                    enhanced_prompt = f"""
                    Based on the medical query: "{query}"
                    
                    Lambda-generated insights:
                    {lambda_response}
                    
                    Additional context from J&J 2025 Workbook:
                    {' '.join(local_contexts[:3])}  # Use top 3 local contexts
                    
                    Please provide a comprehensive medical response that integrates both sources.
                    """
                    
                    # Generate final response using Bedrock with enhanced context
                    final_response = self.generate_medical_response(query, [enhanced_prompt])
                    
                    return {
                        'success': True,
                        'response': final_response['response'],
                        'confidence': final_response['confidence'],
                        'sources': ['lambda_function', 'local_document'],
                        'lambda_contribution': lambda_response,
                        'local_contexts': local_contexts
                    }
                else:
                    # Use Lambda response as primary if no local context
                    return {
                        'success': True,
                        'response': lambda_response,
                        'confidence': 0.8,  # Default confidence for Lambda responses
                        'sources': ['lambda_function'],
                        'lambda_contribution': lambda_response,
                        'local_contexts': []
                    }
            else:
                # Lambda failed, fall back to local processing only
                logger.warning("⚠️ Lambda processing failed, using local processing only")
                return {
                    'success': True,
                    'response': None,  # Signal to use local processing
                    'sources': ['local_document'],
                    'lambda_error': lambda_result.get('error', 'Unknown error')
                }
                
        except Exception as e:
            logger.error(f"❌ Enhanced response processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'sources': []
            }
    
    @log_function_call
    def process_medical_query(self, query: str, therapeutic_area: str = None, include_audio: bool = False, use_lambda: bool = True) -> Dict[str, Any]:
        """Enhanced medical query processing with document-based RAG and Lambda integration"""
        
        logger.info(f"🔍 Processing medical query: '{query[:100]}...'")
        logger.debug(f"Therapeutic area: {therapeutic_area}")
        logger.debug(f"Use Lambda: {use_lambda}")
        
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
                    'therapeutic_area': self.current_therapeutic_area,
                    'sources': ['emergency_detection']
                }
            
            # Context retrieval from processed documents
            logger.debug("📚 Retrieving medical context from documents...")
            context_data = self.retrieve_medical_context(query)
            local_contexts = context_data['contexts']
            
            # Enhanced processing with Lambda integration
            if use_lambda:
                logger.debug("🔄 Using enhanced processing with Lambda integration...")
                
                enhanced_result = self.get_enhanced_response_with_lambda(
                    query, local_contexts, therapeutic_area
                )
                
                if enhanced_result['success'] and enhanced_result.get('response'):
                    # Lambda integration successful
                    processing_time = (datetime.now() - start_time).total_seconds()
                    
                    final_response = {
                        'success': True,
                        'response': enhanced_result['response'],
                        'confidence': enhanced_result.get('confidence', 0.8),
                        'contexts': local_contexts,
                        'detailed_contexts': context_data.get('detailed_contexts', []),
                        'emergency': False,
                        'risk_level': emergency_check.get('risk_level', 'low'),
                        'processing_time': processing_time,
                        'therapeutic_area': self.current_therapeutic_area,
                        'sources': enhanced_result.get('sources', []),
                        'lambda_contribution': enhanced_result.get('lambda_contribution'),
                        'metadata': {
                            'processing_method': 'lambda_enhanced',
                            'lambda_used': True,
                            'document_source': 'J&J_2025_Workbook.pdf',
                            **context_data.get('metadata', {})
                        }
                    }
                    
                    logger.info(f"✅ Enhanced query processed successfully in {processing_time:.3f}s")
                    self._update_conversation_context(query, final_response['response'])
                    return final_response
                
                else:
                    # Lambda failed, fall back to local processing
                    logger.warning("⚠️ Lambda processing failed, falling back to local processing")
                    lambda_error = enhanced_result.get('lambda_error', 'Unknown error')
            
            else:
                logger.debug("📝 Using local processing only (Lambda disabled)")
                lambda_error = "Lambda integration disabled"
            
            # Local processing (fallback or primary)
            logger.debug("🤖 Generating medical response with local processing...")
            response_data = self.generate_medical_response(query, local_contexts)
            
            # Add conversation to context for future queries
            self._update_conversation_context(query, response_data['response'])
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare comprehensive response
            final_response = {
                'success': True,
                'response': response_data['response'],
                'confidence': response_data['confidence'],
                'contexts': local_contexts,
                'detailed_contexts': context_data.get('detailed_contexts', []),
                'emergency': False,
                'risk_level': emergency_check.get('risk_level', 'low'),
                'processing_time': processing_time,
                'therapeutic_area': self.current_therapeutic_area,
                'sources': ['local_document', 'bedrock_llm'],
                'metadata': {
                    **response_data.get('response_metadata', {}),
                    **context_data.get('metadata', {}),
                    'query_enhancement': bool(self.current_therapeutic_area),
                    'document_source': 'J&J_2025_Workbook.pdf',
                    'processing_method': 'local_only',
                    'lambda_used': False,
                    'lambda_error': lambda_error if 'lambda_error' in locals() else None
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
                'therapeutic_area': self.current_therapeutic_area,
                'sources': [],
                'metadata': {'processing_method': 'failed'}
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
    
    @log_function_call
    def get_lambda_metrics(self) -> Dict[str, Any]:
        """Get Lambda function performance metrics"""
        
        if not LAMBDA_MONITORING_AVAILABLE:
            return {'monitoring_available': False, 'message': 'Lambda monitoring not available'}
        
        try:
            # Get metrics for all Lambda functions
            all_metrics = lambda_monitor.get_metrics()
            performance_summary = lambda_monitor.get_performance_summary()
            recent_calls = lambda_monitor.get_recent_calls(minutes=30)
            
            # Focus on wonderscribeconnectVDB function
            wonderscribe_metrics = lambda_monitor.get_metrics('wonderscribeconnectVDB')
            wonderscribe_direct_metrics = lambda_monitor.get_metrics('wonderscribeconnectVDB_direct')
            
            return {
                'monitoring_available': True,
                'summary': performance_summary,
                'wonderscribe_function': wonderscribe_metrics,
                'wonderscribe_direct': wonderscribe_direct_metrics,
                'all_functions': all_metrics,
                'recent_calls': recent_calls[:10],  # Last 10 calls
                'health_status': wonderscribe_metrics.get('health_status', 'unknown') if wonderscribe_metrics else 'unknown'
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get Lambda metrics: {e}")
            return {'monitoring_available': True, 'error': str(e)}
    
    @log_function_call
    def export_lambda_metrics(self, filepath: str = None) -> Optional[str]:
        """Export Lambda metrics to file"""
        
        if not LAMBDA_MONITORING_AVAILABLE:
            logger.warning("⚠️ Lambda monitoring not available for export")
            return None
        
        try:
            if not filepath:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"exports/lambda_metrics_{timestamp}.json"
            
            # Ensure exports directory exists
            os.makedirs("exports", exist_ok=True)
            
            success = lambda_monitor.export_metrics(filepath)
            
            if success:
                logger.info(f"📊 Lambda metrics exported to: {filepath}")
                return filepath
            else:
                logger.error("❌ Failed to export Lambda metrics")
                return None
                
        except Exception as e:
            logger.error(f"❌ Lambda metrics export failed: {e}")
            return None
        """Enhanced AWS connection test with comprehensive validation"""
        logger.debug("🧪 Testing enhanced AWS connection...")
        
        try:
            # Test S3 connection by checking if our document exists
            if self.s3_client:
                logger.debug("📡 Testing S3 connection and document access...")
                try:
                    # Check if the specific document exists
                    self.s3_client.head_object(Bucket=self.s3_bucket, Key=self.s3_document_key)
                    logger.debug("✅ Target document found in S3")
                except Exception as e:
                    logger.warning(f"⚠️ Target document not accessible: {e}")
                
                # Test basic S3 access
                buckets = self.s3_client.list_buckets()
                bucket_count = len(buckets['Buckets'])
                logger.debug(f"S3 test successful: {bucket_count} buckets")
            
            # Test Bedrock connection
            if self.bedrock_client:
                logger.debug("📡 Testing Bedrock connection...")
                # Connection test would go here - Bedrock runtime is ready if client initializes
                pass
            
            # Test Lambda connection if configured
            if self.lambda_client:
                logger.debug("📡 Testing Lambda connection...")
                try:
                    # List functions to test connection
                    self.lambda_client.list_functions(MaxItems=1)
                    logger.debug("✅ Lambda connection successful")
                except Exception as e:
                    logger.warning(f"⚠️ Lambda connection limited: {e}")
            
            if all([self.s3_client, self.bedrock_client]):
                logger.info("✅ Enhanced AWS connection test successful")
                logger.bind(category="aws").info("Core AWS services connected successfully")
                return True
            else:
                logger.error("❌ Some core AWS clients not initialized")
                return False
                
        except Exception as e:
            logger.error(f"❌ Enhanced AWS connection test failed: {e}")
            logger.bind(category="aws").error(f"Connection test failed: {str(e)}")
            return False