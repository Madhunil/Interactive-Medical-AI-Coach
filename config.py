import os
from dotenv import load_dotenv

load_dotenv()

# AWS Configuration
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
BEDROCK_MODEL_ID = os.getenv('BEDROCK_MODEL_ID', 'anthropic.claude-3-haiku-20240307-v1:0')
KNOWLEDGE_BASE_ID = os.getenv('KNOWLEDGE_BASE_ID', 'BDHRZZXGMQ')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'wonderstorytexttoaudiofile')

# Enhanced Medical Configuration
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.7'))
MAX_CONTEXT_SOURCES = int(os.getenv('MAX_CONTEXT_SOURCES', '5'))
SESSION_TIMEOUT_MINUTES = int(os.getenv('SESSION_TIMEOUT_MINUTES', '60'))

# Therapeutic Areas Configuration for Medical AI Coach
THERAPEUTIC_AREAS = {
    'oncology': {
        'name': 'Oncology',
        'icon': '🎗️',
        'description': 'Cancer treatment, immunotherapy, targeted therapies, and oncological care',
        'knowledge_base_prefix': 'oncology',
        'emergency_keywords': [
            'tumor lysis syndrome', 'neutropenic fever', 'severe mucositis',
            'cytokine release syndrome', 'severe nausea vomiting', 'febrile neutropenia'
        ],
        'quick_modules': [
            'Immunotherapy Mechanisms',
            'Targeted Therapy Selection',
            'Managing Adverse Events',
            'Biomarker Testing',
            'Cancer Staging'
        ],
        'common_questions': [
            'How does immunotherapy work in cancer treatment?',
            'What are the side effects of chemotherapy?',
            'How to manage treatment-related adverse events?',
            'When to use combination cancer therapy?'
        ]
    },
    'neuroscience': {
        'name': 'Neuroscience',
        'icon': '🧠',
        'description': 'CNS disorders, neurodegenerative diseases, and psychiatric conditions',
        'knowledge_base_prefix': 'neuroscience',
        'emergency_keywords': [
            'suicidal ideation', 'serotonin syndrome', 'neuroleptic malignant syndrome',
            'status epilepticus', 'severe depression', 'psychotic episode'
        ],
        'quick_modules': [
            'Neurodegenerative Diseases',
            'Depression Treatment',
            'Seizure Management',
            'Cognitive Assessment',
            'Psychiatric Emergencies'
        ],
        'common_questions': [
            'What are the latest treatments for Alzheimer\'s disease?',
            'How to manage treatment-resistant depression?',
            'What are the signs of serotonin syndrome?',
            'When to switch antidepressants?'
        ]
    },
    'immunology': {
        'name': 'Immunology',
        'icon': '🛡️',
        'description': 'Autoimmune diseases, inflammatory conditions, and immunological disorders',
        'knowledge_base_prefix': 'immunology',
        'emergency_keywords': [
            'severe allergic reaction', 'anaphylaxis', 'cytokine storm',
            'severe infection', 'immunosuppression complications'
        ],
        'quick_modules': [
            'Autoimmune Mechanisms',
            'Immunosuppressive Therapy',
            'Allergy Management',
            'Vaccination Guidelines',
            'Immune System Disorders'
        ],
        'common_questions': [
            'How do autoimmune diseases develop?',
            'What are the risks of immunosuppressive therapy?',
            'How to manage severe allergic reactions?',
            'When to use biological therapies?'
        ]
    },
    'cardiovascular': {
        'name': 'Cardiovascular',
        'icon': '❤️',
        'description': 'Heart disease, hypertension, lipid disorders, and cardiovascular conditions',
        'knowledge_base_prefix': 'cardiovascular',
        'emergency_keywords': [
            'chest pain', 'heart attack', 'stroke', 'pulmonary embolism',
            'severe hypertension', 'heart failure exacerbation'
        ],
        'quick_modules': [
            'Heart Failure Management',
            'Acute Coronary Syndromes',
            'Hypertension Treatment',
            'Lipid Management',
            'Cardiac Emergencies'
        ],
        'common_questions': [
            'How to manage acute heart failure?',
            'What are the new cholesterol guidelines?',
            'How to treat hypertensive crisis?',
            'When to use anticoagulation therapy?'
        ]
    },
    'pulmonary': {
        'name': 'Pulmonary Medicine',
        'icon': '🫁',
        'description': 'Respiratory disorders, pulmonary diseases, and breathing conditions',
        'knowledge_base_prefix': 'pulmonary',
        'emergency_keywords': [
            'respiratory failure', 'severe asthma', 'pneumothorax',
            'pulmonary embolism', 'acute dyspnea'
        ],
        'quick_modules': [
            'Asthma Management',
            'COPD Treatment',
            'Pulmonary Embolism',
            'Respiratory Failure',
            'Lung Cancer Screening'
        ],
        'common_questions': [
            'How to manage acute asthma exacerbation?',
            'What are the signs of pulmonary embolism?',
            'How to treat COPD exacerbation?',
            'When to intubate a patient?'
        ]
    },
    'infectious_diseases': {
        'name': 'Infectious Diseases',
        'icon': '🦠',
        'description': 'Antimicrobial therapy, infections, and infectious disease management',
        'knowledge_base_prefix': 'infectious',
        'emergency_keywords': [
            'sepsis', 'severe infection', 'antibiotic resistance',
            'meningitis', 'necrotizing fasciitis'
        ],
        'quick_modules': [
            'Antimicrobial Stewardship',
            'Sepsis Management',
            'Antibiotic Resistance',
            'Infection Control',
            'Viral Infections'
        ],
        'common_questions': [
            'How to manage severe sepsis?',
            'What antibiotics for MRSA infection?',
            'How to prevent healthcare-associated infections?',
            'When to use combination antibiotics?'
        ]
    },
    'emergency_medicine': {
        'name': 'Emergency Medicine',
        'icon': '🚨',
        'description': 'Emergency care, trauma management, and critical situations',
        'knowledge_base_prefix': 'emergency',
        'emergency_keywords': [
            'cardiac arrest', 'trauma', 'shock', 'respiratory arrest',
            'severe bleeding', 'unconscious patient'
        ],
        'quick_modules': [
            'ACLS Protocols',
            'Trauma Assessment',
            'Shock Management',
            'Emergency Procedures',
            'Triage Principles'
        ],
        'common_questions': [
            'How to manage cardiac arrest?',
            'What is the trauma assessment protocol?',
            'How to treat different types of shock?',
            'When to perform emergency procedures?'
        ]
    },
    'pharmacology': {
        'name': 'Pharmacology',
        'icon': '💊',
        'description': 'Drug mechanisms, interactions, and pharmaceutical therapy',
        'knowledge_base_prefix': 'pharmacology',
        'emergency_keywords': [
            'drug overdose', 'severe adverse reaction', 'drug interaction',
            'anaphylaxis', 'toxicity'
        ],
        'quick_modules': [
            'Drug Mechanisms',
            'Drug Interactions',
            'Adverse Reactions',
            'Pharmacokinetics',
            'Therapeutic Monitoring'
        ],
        'common_questions': [
            'How do ACE inhibitors work?',
            'What are common drug interactions?',
            'How to manage drug overdose?',
            'When to adjust drug dosages?'
        ]
    }
}

# Response Configuration
RESPONSE_TEMPLATES = {
    'emergency': {
        'prefix': '🚨 MEDICAL EMERGENCY DETECTED 🚨\n\n',
        'action': 'Please seek immediate medical attention or call emergency services (911).',
        'suffix': '\n\nThis is an automated safety alert based on keywords in your question.'
    },
    'high_risk': {
        'prefix': '⚠️ HIGH-RISK SCENARIO ⚠️\n\n',
        'action': 'This situation requires immediate evaluation by qualified medical professionals.',
        'suffix': '\n\nPlease consult with healthcare professionals or seek medical attention.'
    },
    'standard': {
        'prefix': '',
        'action': '',
        'suffix': '\n\n💡 Remember: This information is for educational purposes only. Always follow current clinical guidelines and consult with healthcare professionals for patient care decisions.'
    }
}

# Enhanced Medical Disclaimer
MEDICAL_DISCLAIMER = """
⚠️ IMPORTANT MEDICAL DISCLAIMER - INTERACTIVE MEDICAL AI COACH:

• This AI system is designed for medical education and training purposes only
• Information provided is based on medical literature and current guidelines
• AI-generated content may contain errors and should not be relied upon as definitive medical guidance
• All medical decisions must be made by qualified healthcare professionals
• Always consult current clinical guidelines, drug prescribing information, and institutional protocols
• In emergencies, contact emergency services (911) or your local emergency response team immediately
• This system cannot diagnose, treat, cure, or prevent any disease
• Individual patient care requires comprehensive evaluation by qualified clinicians

FOR EDUCATIONAL USE ONLY - NOT FOR PATIENT CARE DECISIONS
"""

# Audio Configuration
AUDIO_SETTINGS = {
    'max_recording_duration': 30,  # seconds
    'supported_formats': ['wav', 'mp3', 'flac'],
    'transcription_confidence_threshold': 0.8,
    'enable_tts': True,
    'default_voice': 'Joanna',
    'speech_rate': 1.0
}

# Knowledge Base Configuration per Therapeutic Area
KNOWLEDGE_BASE_CONFIG = {
    area_key: {
        'knowledge_base_id': f"{KNOWLEDGE_BASE_ID}_{area_key}",
        'max_results': 5,
        'similarity_threshold': 0.7,
        'context_window': 2000
    }
    for area_key in THERAPEUTIC_AREAS.keys()
}

# Session Analytics Configuration
SESSION_ANALYTICS = {
    'track_interactions': True,
    'calculate_confidence': True,
    'export_sessions': True,
    'anonymize_data': True
}

# Emergency Response Configuration
EMERGENCY_RESPONSE = {
    'immediate_alert': True,
    'documentation_required': True,
    'safety_protocols': True
}

# System Configuration
SYSTEM_CONFIG = {
    'max_concurrent_sessions': 50,
    'session_timeout_minutes': 60,
    'auto_save_enabled': True,
    'performance_monitoring': True
}

# UI Configuration
UI_CONFIG = {
    'theme': 'modern_medical',
    'show_confidence_scores': True,
    'enable_quick_actions': True,
    'sidebar_expanded': True,
    'chat_history_limit': 50
}