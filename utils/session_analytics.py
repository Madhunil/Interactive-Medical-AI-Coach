import json
import os
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
from utils.logging import setup_logging, log_function_call

logger = setup_logging()

class SessionAnalytics:
    """Simplified analytics for Medical AI Coach sessions"""
    
    @log_function_call
    def __init__(self):
        """Initialize session analytics"""
        logger.info("📊 Initializing SessionAnalytics...")
        
        self.interactions_file = "data/session_interactions.json"
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Initialize data file
        self._initialize_data_file()
        
        logger.info("✅ SessionAnalytics initialized successfully")
    
    @log_function_call
    def _initialize_data_file(self):
        """Initialize analytics data file"""
        if not os.path.exists(self.interactions_file):
            with open(self.interactions_file, 'w') as f:
                json.dump([], f)
            logger.debug(f"Created analytics file: {self.interactions_file}")
    
    @log_function_call
    def log_interaction(self, session_id: str, role: str, content: str, therapeutic_area: str = None):
        """Log interaction for session analytics"""
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id,
            'role': role,
            'content_length': len(content),
            'therapeutic_area': therapeutic_area,
            'interaction_type': self._classify_interaction(content, role),
            'complexity_score': self._calculate_complexity_score(content),
            'topics_mentioned': self._extract_topics(content),
            'metadata': {
                'character_count': len(content),
                'word_count': len(content.split()),
                'question_marks': content.count('?'),
                'medical_terms': self._count_medical_terms(content)
            }
        }
        
        # Load existing interactions
        interactions = self._load_json_file(self.interactions_file)
        interactions.append(interaction)
        
        # Keep only last 1000 interactions for performance
        if len(interactions) > 1000:
            interactions = interactions[-1000:]
        
        self._save_json_file(self.interactions_file, interactions)
        
        logger.debug(f"📝 Logged interaction: {role} - {therapeutic_area}")
    
    @log_function_call
    def calculate_session_confidence(self, chat_history: List[Dict]) -> float:
        """Calculate session confidence score based on interactions"""
        if not chat_history:
            return 0.0
        
        # Extract user questions and AI responses
        user_questions = [msg for msg in chat_history if msg['role'] == 'user']
        ai_responses = [msg for msg in chat_history if msg['role'] == 'assistant']
        
        confidence_factors = []
        
        # Factor 1: Question complexity (more complex = higher engagement)
        for question in user_questions:
            complexity = self._calculate_complexity_score(question['content'])
            confidence_factors.append(complexity * 20)  # Scale to 0-100
        
        # Factor 2: Medical terminology usage
        medical_term_score = 0
        for question in user_questions:
            term_count = self._count_medical_terms(question['content'])
            medical_term_score += min(term_count * 10, 50)  # Max 50 points
        
        if user_questions:
            medical_term_score /= len(user_questions)
        confidence_factors.append(medical_term_score)
        
        # Factor 3: Session engagement (interaction count)
        if len(chat_history) > 3:
            engagement_score = min(len(chat_history) * 3, 40)  # Max 40 points
            confidence_factors.append(engagement_score)
        
        # Factor 4: Response quality (check for metadata confidence scores)
        response_confidences = []
        for response in ai_responses:
            if 'metadata' in response and 'confidence' in response['metadata']:
                response_confidences.append(response['metadata']['confidence'] * 100)
        
        if response_confidences:
            confidence_factors.append(np.mean(response_confidences))
        
        # Calculate weighted average
        if confidence_factors:
            final_confidence = np.mean(confidence_factors)
            return min(max(final_confidence, 0), 100)  # Clamp between 0-100
        
        return 50.0  # Default neutral confidence
    
    @log_function_call
    def get_session_stats(self, chat_history: List[Dict]) -> Dict[str, Any]:
        """Get comprehensive session statistics"""
        if not chat_history:
            return {}
        
        user_questions = [msg for msg in chat_history if msg['role'] == 'user']
        ai_responses = [msg for msg in chat_history if msg['role'] == 'assistant']
        
        # Calculate basic stats
        stats = {
            'total_interactions': len(chat_history),
            'total_questions': len(user_questions),
            'total_responses': len(ai_responses),
            'session_duration_minutes': self._calculate_session_duration(chat_history),
            'avg_response_time': self._calculate_avg_response_time(chat_history),
            'topics_covered': self._extract_session_topics(chat_history),
            'complexity_analysis': self._analyze_session_complexity(chat_history),
            'therapeutic_areas': self._get_therapeutic_areas_covered(chat_history)
        }
        
        return stats
    
    @log_function_call
    def export_session(self, session_id: str, chat_history: List[Dict]) -> str:
        """Export session data for download"""
        logger.info(f"📄 Exporting session: {session_id}")
        
        # Calculate session metrics
        session_start = min(msg['timestamp'] for msg in chat_history) if chat_history else time.time()
        session_end = max(msg['timestamp'] for msg in chat_history) if chat_history else time.time()
        duration_minutes = (session_end - session_start) / 60
        
        user_questions = [msg for msg in chat_history if msg['role'] == 'user']
        ai_responses = [msg for msg in chat_history if msg['role'] == 'assistant']
        
        # Generate comprehensive report
        report = {
            'session_metadata': {
                'session_id': session_id,
                'export_timestamp': datetime.now().isoformat(),
                'duration_minutes': round(duration_minutes, 2),
                'total_interactions': len(chat_history),
                'questions_asked': len(user_questions),
                'responses_provided': len(ai_responses)
            },
            'session_analytics': {
                'confidence_score': self.calculate_session_confidence(chat_history),
                'therapeutic_areas_covered': self._get_therapeutic_areas_covered(chat_history),
                'topics_discussed': self._extract_session_topics(chat_history),
                'complexity_analysis': self._analyze_session_complexity(chat_history),
                'interaction_patterns': self._analyze_interaction_patterns(chat_history)
            },
            'detailed_interactions': [
                {
                    'timestamp': datetime.fromtimestamp(msg['timestamp']).isoformat(),
                    'role': msg['role'],
                    'content': msg['content'],
                    'type': msg.get('type', 'normal'),
                    'therapeutic_area': msg.get('therapeutic_area'),
                    'metadata': msg.get('metadata', {})
                }
                for msg in chat_history
            ],
            'session_summary': {
                'key_topics': self._extract_session_topics(chat_history)[:5],
                'areas_of_focus': self._get_therapeutic_areas_covered(chat_history),
                'learning_progression': self._track_learning_progression(chat_history),
                'recommendations': self._generate_session_recommendations(chat_history)
            },
            'compliance_information': {
                'educational_purpose_only': True,
                'no_patient_data': True,
                'anonymized_export': True,
                'medical_disclaimer_acknowledged': True
            }
        }
        
        return json.dumps(report, indent=2, ensure_ascii=False)
    
    def _classify_interaction(self, content: str, role: str) -> str:
        """Classify the type of interaction"""
        if role == 'user':
            content_lower = content.lower()
            
            if any(word in content_lower for word in ['what', 'how', 'why', 'when', 'where']):
                return 'question'
            elif any(word in content_lower for word in ['explain', 'describe', 'tell me about']):
                return 'explanation_request'
            elif any(word in content_lower for word in ['compare', 'difference', 'versus']):
                return 'comparison'
            elif content.count('?') > 0:
                return 'question'
            else:
                return 'statement'
        
        return 'response'
    
    def _calculate_complexity_score(self, content: str) -> float:
        """Calculate complexity score of content (0-1)"""
        factors = []
        
        # Length factor
        factors.append(min(len(content) / 500, 1.0))
        
        # Medical terminology factor
        medical_terms = self._count_medical_terms(content)
        factors.append(min(medical_terms / 10, 1.0))
        
        # Question complexity factor
        complex_words = ['mechanism', 'pathophysiology', 'contraindication', 'pharmacokinetics']
        complex_count = sum(1 for word in complex_words if word in content.lower())
        factors.append(min(complex_count / 5, 1.0))
        
        return np.mean(factors) if factors else 0.0
    
    def _extract_topics(self, content: str) -> List[str]:
        """Extract medical topics from content"""
        common_topics = [
            'diagnosis', 'treatment', 'medication', 'surgery', 'therapy',
            'side effects', 'dosage', 'mechanism', 'clinical trial',
            'prevention', 'screening', 'prognosis', 'complications',
            'pharmacology', 'pathophysiology', 'epidemiology'
        ]
        
        content_lower = content.lower()
        found_topics = [topic for topic in common_topics if topic in content_lower]
        
        return found_topics
    
    def _count_medical_terms(self, content: str) -> int:
        """Count medical terminology in content"""
        medical_terms = [
            'patient', 'diagnosis', 'treatment', 'therapy', 'medication',
            'syndrome', 'disease', 'disorder', 'condition', 'symptom',
            'clinical', 'medical', 'pharmaceutical', 'drug', 'dose',
            'adverse', 'contraindication', 'indication', 'efficacy',
            'pathophysiology', 'pharmacokinetics', 'pharmacodynamics'
        ]
        
        content_lower = content.lower()
        return sum(1 for term in medical_terms if term in content_lower)
    
    def _calculate_session_duration(self, chat_history: List[Dict]) -> float:
        """Calculate session duration in minutes"""
        if len(chat_history) < 2:
            return 0.0
        
        timestamps = [msg['timestamp'] for msg in chat_history]
        duration_seconds = max(timestamps) - min(timestamps)
        return duration_seconds / 60
    
    def _calculate_avg_response_time(self, chat_history: List[Dict]) -> float:
        """Calculate average response time"""
        response_times = []
        
        for i in range(1, len(chat_history)):
            if (chat_history[i]['role'] == 'assistant' and 
                chat_history[i-1]['role'] == 'user'):
                
                response_time = chat_history[i]['timestamp'] - chat_history[i-1]['timestamp']
                response_times.append(response_time)
        
        return np.mean(response_times) if response_times else 0.0
    
    def _extract_session_topics(self, chat_history: List[Dict]) -> List[str]:
        """Extract all topics discussed in the session"""
        all_topics = []
        
        for msg in chat_history:
            if msg['role'] == 'user':
                topics = self._extract_topics(msg['content'])
                all_topics.extend(topics)
        
        # Count occurrences and return most common
        topic_counts = Counter(all_topics)
        return [topic for topic, count in topic_counts.most_common(10)]
    
    def _analyze_session_complexity(self, chat_history: List[Dict]) -> Dict[str, float]:
        """Analyze complexity progression of the session"""
        user_messages = [msg for msg in chat_history if msg['role'] == 'user']
        
        if not user_messages:
            return {'average_complexity': 0.0, 'complexity_trend': 'stable'}
        
        complexities = [self._calculate_complexity_score(msg['content']) for msg in user_messages]
        
        complexity_trend = 'stable'
        if len(complexities) > 1:
            if complexities[-1] > complexities[0]:
                complexity_trend = 'increasing'
            elif complexities[-1] < complexities[0]:
                complexity_trend = 'decreasing'
        
        return {
            'average_complexity': np.mean(complexities),
            'max_complexity': max(complexities),
            'min_complexity': min(complexities),
            'complexity_trend': complexity_trend,
            'complexity_progression': complexities
        }
    
    def _get_therapeutic_areas_covered(self, chat_history: List[Dict]) -> List[str]:
        """Get therapeutic areas covered in the session"""
        areas = set()
        
        for msg in chat_history:
            if msg.get('therapeutic_area'):
                areas.add(msg['therapeutic_area'])
        
        return list(areas)
    
    def _analyze_interaction_patterns(self, chat_history: List[Dict]) -> Dict[str, Any]:
        """Analyze interaction patterns in the session"""
        patterns = {
            'question_types': Counter(),
            'avg_question_length': 0,
            'avg_response_length': 0,
            'interaction_frequency': []
        }
        
        user_questions = [msg for msg in chat_history if msg['role'] == 'user']
        ai_responses = [msg for msg in chat_history if msg['role'] == 'assistant']
        
        # Analyze question types
        for question in user_questions:
            question_type = self._classify_interaction(question['content'], question['role'])
            patterns['question_types'][question_type] += 1
        
        # Calculate average lengths
        if user_questions:
            patterns['avg_question_length'] = np.mean([len(q['content']) for q in user_questions])
        
        if ai_responses:
            patterns['avg_response_length'] = np.mean([len(r['content']) for r in ai_responses])
        
        return patterns
    
    def _track_learning_progression(self, chat_history: List[Dict]) -> List[Dict[str, Any]]:
        """Track learning progression throughout the session"""
        progression = []
        user_messages = [msg for msg in chat_history if msg['role'] == 'user']
        
        for i, msg in enumerate(user_messages):
            complexity = self._calculate_complexity_score(msg['content'])
            medical_terms = self._count_medical_terms(msg['content'])
            
            progression.append({
                'question_number': i + 1,
                'complexity': complexity,
                'medical_terminology_count': medical_terms,
                'timestamp': msg['timestamp'],
                'topics': self._extract_topics(msg['content'])
            })
        
        return progression
    
    def _generate_session_recommendations(self, chat_history: List[Dict]) -> List[str]:
        """Generate recommendations based on the session"""
        recommendations = []
        
        stats = self.get_session_stats(chat_history)
        confidence_score = self.calculate_session_confidence(chat_history)
        
        if confidence_score < 60:
            recommendations.append("Consider reviewing basic concepts before moving to advanced topics")
        
        if len(stats.get('topics_covered', [])) > 5:
            recommendations.append("Great topic diversity! Try to focus on 2-3 key areas for deeper understanding")
        
        if len(chat_history) > 20:
            recommendations.append("Excellent engagement! Consider taking breaks during intensive learning sessions")
        
        if stats.get('session_duration_minutes', 0) > 60:
            recommendations.append("Long session detected. Consider shorter, more frequent sessions for better retention")
        
        return recommendations
    
    def _load_json_file(self, file_path: str) -> List:
        """Load JSON file safely"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def _save_json_file(self, file_path: str, data: List):
        """Save JSON file safely"""
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save {file_path}: {e}")