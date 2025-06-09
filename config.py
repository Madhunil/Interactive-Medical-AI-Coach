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

# Medical Disclaimer
MEDICAL_DISCLAIMER = """
⚠️ IMPORTANT MEDICAL DISCLAIMER:
• This AI system provides general health information only
• AI-generated content may contain errors and should not be relied upon as medical guidance
• Always consult qualified healthcare professionals for medical advice
• In emergencies, contact 911 or your local emergency services immediately
• This system cannot diagnose, treat, cure, or prevent any disease
"""