"""
📝 FORM SERVICE FOR PHARMACOVIGILANCE FOLLOW-UP

This module manages patient follow-up forms - creating, tracking, and
processing form submissions.

Author: Generated for dual-channel communication feature
"""

import os
import json
import logging
import secrets
from typing import Dict, Any, List, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

FORM_BASE_URL = os.getenv('FORM_BASE_URL', 'http://localhost:8000/form')
FORM_SECRET_KEY = os.getenv('FORM_SECRET_KEY', 'your-secret-key-change-in-production')


# =============================================================================
# FORM QUESTIONS DEFINITION
# =============================================================================

# All questions in multiple languages
FORM_QUESTIONS = {
    'medicine_started': {
        'id': 'Q2_medicine_started',
        'type': 'radio',
        'required': True,
        'labels': {
            'en': 'Have you started taking the prescribed medicine?',
            'hi': 'क्या आपने निर्धारित दवा लेना शुरू कर दिया है?',
            'ta': 'நீங்கள் பரிந்துரைக்கப்பட்ட மருந்தை எடுக்க ஆரம்பித்தீர்களா?',
            'te': 'మీరు సూచించిన మందులు తీసుకోవడం ప్రారంభించారా?',
            'ml': 'നിർദ്ദേശിച്ച മരുന്ന് കഴിക്കാൻ തുടങ്ങിയോ?'
        },
        'options': {
            'en': ['Yes, I have started', 'No, not yet', 'I will start tomorrow'],
            'hi': ['हां, मैंने शुरू कर दिया है', 'नहीं, अभी नहीं', 'मैं कल शुरू करूंगा'],
            'ta': ['ஆம், நான் தொடங்கிவிட்டேன்', 'இல்லை, இன்னும் இல்லை', 'நாளை தொடங்குவேன்'],
            'te': ['అవును, నేను ప్రారంభించాను', 'లేదు, ఇంకా లేదు', 'రేపు మొదలు పెడతాను'],
            'ml': ['അതെ, ഞാൻ തുടങ്ങി', 'ഇല്ല, ഇതുവരെ', 'ഞാൻ നാളെ തുടങ്ങും']
        },
        'values': ['yes', 'no', 'tomorrow']
    },
    'adherence': {
        'id': 'Q3_adherence',
        'type': 'radio',
        'required': True,
        'labels': {
            'en': 'How often are you taking the medicine daily?',
            'hi': 'आप दवा रोजाना कितनी बार ले रहे हैं?',
            'ta': 'தினமும் எத்தனை முறை மருந்து எடுக்கிறீர்கள்?',
            'te': 'రోజూ ఎన్నిసార్లు మందులు తీసుకుంటున్నారు?',
            'ml': 'ദിവസവും എത്ര തവണ മരുന്ന് കഴിക്കുന്നു?'
        },
        'options': {
            'en': ['Once a day', 'Twice a day', 'Three times a day', 'As needed'],
            'hi': ['दिन में एक बार', 'दिन में दो बार', 'दिन में तीन बार', 'जरूरत के अनुसार'],
            'ta': ['ஒரு நாளைக்கு ஒரு முறை', 'ஒரு நாளைக்கு இரண்டு முறை', 'ஒரு நாளைக்கு மூன்று முறை', 'தேவைப்படும்போது'],
            'te': ['రోజుకు ఒకసారి', 'రోజుకు రెండుసార్లు', 'రోజుకు మూడు సార్లు', 'అవసరమైనప్పుడు'],
            'ml': ['ദിവസത്തിൽ ഒരിക്കൽ', 'ദിവസത്തിൽ രണ്ടുതവണ', 'ദിവസത്തിൽ മൂന്നുതവണ', 'ആവശ്യാനുസരണം']
        },
        'values': ['once', 'twice', 'thrice', 'as_needed']
    },
    'food_relation': {
        'id': 'Q4_food_relation',
        'type': 'radio',
        'required': True,
        'labels': {
            'en': 'When do you take the medicine in relation to food?',
            'hi': 'भोजन के संबंध में आप दवा कब लेते हैं?',
            'ta': 'உணவுடன் தொடர்புடைய மருந்தை எப்போது எடுக்கிறீர்கள்?',
            'te': 'ఆహారానికి సంబంధించి మందులు ఎప్పుడు తీసుకుంటారు?',
            'ml': 'ഭക്ഷണവുമായി ബന്ധപ്പെട്ട് മരുന്ന് എപ്പോൾ കഴിക്കുന്നു?'
        },
        'options': {
            'en': ['Before food', 'After food', 'With food', 'Empty stomach'],
            'hi': ['भोजन से पहले', 'भोजन के बाद', 'भोजन के साथ', 'खाली पेट'],
            'ta': ['உணவுக்கு முன்', 'உணவுக்குப் பின்', 'உணவுடன்', 'வெறும் வயிற்றில்'],
            'te': ['ఆహారానికి ముందు', 'ఆహారం తర్వాత', 'ఆహారంతో', 'ఖాళీ కడుపుతో'],
            'ml': ['ഭക്ഷണത്തിന് മുമ്പ്', 'ഭക്ഷണത്തിന് ശേഷം', 'ഭക്ഷണത്തോടൊപ്പം', 'വെറും വയറ്റിൽ']
        },
        'values': ['before', 'after', 'with', 'empty']
    },
    'overall_feeling': {
        'id': 'Q6_overall_feeling',
        'type': 'radio',
        'required': True,
        'labels': {
            'en': 'Since starting the medicine, how are you feeling overall?',
            'hi': 'दवा शुरू करने के बाद से आप कुल मिलाकर कैसा महसूस कर रहे हैं?',
            'ta': 'மருந்து தொடங்கியதிலிருந்து ஒட்டுமொத்தமாக எப்படி உணர்கிறீர்கள்?',
            'te': 'మందులు మొదలు పెట్టినప్పటి నుండి మీరు ఎలా ఫీల్ అవుతున్నారు?',
            'ml': 'മരുന്ന് കഴിക്കാൻ തുടങ്ങിയതിന് ശേഷം മൊത്തത്തിൽ എങ്ങനെ തോന്നുന്നു?'
        },
        'options': {
            'en': ['Better 😊', 'Same 😐', 'Worse 😔', 'Much worse 😰'],
            'hi': ['बेहतर 😊', 'वही 😐', 'खराब 😔', 'बहुत खराब 😰'],
            'ta': ['சிறப்பாக 😊', 'அதே 😐', 'மோசமாக 😔', 'மிகவும் மோசமாக 😰'],
            'te': ['మెరుగ్గా 😊', 'అదే 😐', 'అధ్వాన్నంగా 😔', 'చాలా అధ్వాన్నంగా 😰'],
            'ml': ['മെച്ചം 😊', 'അതേപോലെ 😐', 'മോശം 😔', 'വളരെ മോശം 😰']
        },
        'values': ['better', 'same', 'worse', 'much_worse']
    },
    'new_symptoms': {
        'id': 'Q7_new_symptoms',
        'type': 'radio',
        'required': True,
        'labels': {
            'en': 'Have you noticed any new symptoms or discomfort?',
            'hi': 'क्या आपने कोई नए लक्षण या असुविधा देखी है?',
            'ta': 'புதிய அறிகுறிகள் அல்லது அசௌகரியத்தை கவனித்தீர்களா?',
            'te': 'కొత్త లక్షణాలు లేదా అసౌకర్యం గమనించారా?',
            'ml': 'പുതിയ ലക്ഷണങ്ങളോ അസ്വസ്ഥതയോ ശ്രദ്ധിച്ചിട്ടുണ്ടോ?'
        },
        'options': {
            'en': ['Yes, I have new symptoms', 'No new symptoms'],
            'hi': ['हां, मुझे नए लक्षण हैं', 'कोई नए लक्षण नहीं'],
            'ta': ['ஆம், புதிய அறிகுறிகள் உள்ளன', 'புதிய அறிகுறிகள் இல்லை'],
            'te': ['అవును, కొత్త లక్షణాలు ఉన్నాయి', 'కొత్త లక్షణాలు లేవు'],
            'ml': ['അതെ, പുതിയ ലക്ഷണങ്ങളുണ്ട്', 'പുതിയ ലക്ഷണങ്ങളില്ല']
        },
        'values': ['yes', 'no']
    },
    'symptom_description': {
        'id': 'Q8_symptom_description',
        'type': 'textarea',
        'required': False,
        'conditional': {'field': 'new_symptoms', 'value': 'yes'},
        'labels': {
            'en': 'Please describe the symptoms you are experiencing:',
            'hi': 'कृपया उन लक्षणों का वर्णन करें जो आप अनुभव कर रहे हैं:',
            'ta': 'நீங்கள் அனுபவிக்கும் அறிகுறிகளை விவரிக்கவும்:',
            'te': 'మీరు అనుభవిస్తున్న లక్షణాలను వివరించండి:',
            'ml': 'നിങ്ങൾ അനുഭവിക്കുന്ന ലക്ഷണങ്ങൾ വിവരിക്കുക:'
        }
    },
    'onset': {
        'id': 'Q9_onset',
        'type': 'radio',
        'required': False,
        'conditional': {'field': 'new_symptoms', 'value': 'yes'},
        'labels': {
            'en': 'When did these symptoms start?',
            'hi': 'ये लक्षण कब शुरू हुए?',
            'ta': 'இந்த அறிகுறிகள் எப்போது தொடங்கின?',
            'te': 'ఈ లక్షణాలు ఎప్పుడు మొదలయ్యాయి?',
            'ml': 'ഈ ലക്ഷണങ്ങൾ എപ്പോൾ ആരംഭിച്ചു?'
        },
        'options': {
            'en': ['After first dose', 'Within 1 day', 'After 2-3 days', 'After more than 3 days'],
            'hi': ['पहली खुराक के बाद', '1 दिन के भीतर', '2-3 दिन बाद', '3 दिन से अधिक के बाद'],
            'ta': ['முதல் டோஸுக்குப் பிறகு', '1 நாளுக்குள்', '2-3 நாட்களுக்குப் பிறகு', '3 நாட்களுக்கு மேல் பிறகு'],
            'te': ['మొదటి డోస్ తర్వాత', '1 రోజులోపు', '2-3 రోజుల తర్వాత', '3 రోజుల తర్వాత'],
            'ml': ['ആദ്യ ഡോസിന് ശേഷം', '1 ദിവസത്തിനുള്ളിൽ', '2-3 ദിവസങ്ങൾക്ക് ശേഷം', '3 ദിവസത്തിന് ശേഷം']
        },
        'values': ['first_dose', 'within_1_day', '2_3_days', 'more_than_3_days']
    },
    'severity': {
        'id': 'Q10_severity',
        'type': 'radio',
        'required': False,
        'conditional': {'field': 'new_symptoms', 'value': 'yes'},
        'labels': {
            'en': 'How severe are the symptoms?',
            'hi': 'लक्षण कितने गंभीर हैं?',
            'ta': 'அறிகுறிகள் எவ்வளவு கடுமையானவை?',
            'te': 'లక్షణాలు ఎంత తీవ్రంగా ఉన్నాయి?',
            'ml': 'ലക്ഷണങ്ങൾ എത്ര കഠിനമാണ്?'
        },
        'options': {
            'en': ['Mild (noticeable but manageable)', 'Moderate (uncomfortable, affecting daily life)', 'Severe (needs medical attention) ⚠️'],
            'hi': ['हल्का (ध्यान देने योग्य लेकिन प्रबंधनीय)', 'मध्यम (असहज, दैनिक जीवन को प्रभावित करता है)', 'गंभीर (चिकित्सा ध्यान की जरूरत है) ⚠️'],
            'ta': ['லேசான (கவனிக்கத்தக்கது ஆனால் சமாளிக்கக்கூடியது)', 'மிதமான (அசௌகரியமான, தினசரி வாழ்க்கையை பாதிக்கிறது)', 'கடுமையான (மருத்துவ கவனிப்பு தேவை) ⚠️'],
            'te': ['తేలికపాటి (గమనించదగినది కానీ నిర్వహించదగినది)', 'మధ్యస్థం (అసౌకర్యం, దైనందిన జీవితాన్ని ప్రభావితం చేస్తుంది)', 'తీవ్రమైన (వైద్య శ్రద్ధ అవసరం) ⚠️'],
            'ml': ['നേരിയ (ശ്രദ്ധേയമാണ് പക്ഷേ കൈകാര്യം ചെയ്യാവുന്നതാണ്)', 'മിതമായ (അസൌകര്യം, ദൈനംദിന ജീവിതത്തെ ബാധിക്കുന്നു)', 'കഠിനമായ (വൈദ്യ ശ്രദ്ധ ആവശ്യമാണ്) ⚠️']
        },
        'values': ['mild', 'moderate', 'severe']
    },
    'body_parts': {
        'id': 'Q11_body_parts',
        'type': 'checkbox',
        'required': False,
        'conditional': {'field': 'new_symptoms', 'value': 'yes'},
        'labels': {
            'en': 'Which part of your body is affected? (Select all that apply)',
            'hi': 'आपके शरीर का कौन सा हिस्सा प्रभावित है? (सभी लागू का चयन करें)',
            'ta': 'உங்கள் உடலின் எந்த பகுதி பாதிக்கப்பட்டுள்ளது? (பொருந்தும் அனைத்தையும் தேர்ந்தெடுக்கவும்)',
            'te': 'మీ శరీరంలో ఏ భాగం ప్రభావితమైంది? (వర్తించే అన్నింటినీ ఎంచుకోండి)',
            'ml': 'നിങ്ങളുടെ ശരീരത്തിന്റെ ഏത് ഭാഗമാണ് ബാധിച്ചത്? (ബാധകമായ എല്ലാം തിരഞ്ഞെടുക്കുക)'
        },
        'options': {
            'en': ['Skin', 'Stomach/Digestive', 'Head', 'Chest', 'Breathing', 'Other'],
            'hi': ['त्वचा', 'पेट/पाचन', 'सिर', 'छाती', 'सांस', 'अन्य'],
            'ta': ['தோல்', 'வயிறு/செரிமானம்', 'தலை', 'மார்பு', 'சுவாசம்', 'மற்றவை'],
            'te': ['చర్మం', 'కడుపు/జీర్ణ', 'తల', 'ఛాతీ', 'శ్వాస', 'ఇతర'],
            'ml': ['ചർമ്മം', 'വയറ്/ദഹനം', 'തല', 'നെഞ്ച്', 'ശ്വസനം', 'മറ്റുള്ളവ']
        },
        'values': ['skin', 'stomach', 'head', 'chest', 'breathing', 'other']
    },
    'safety_confirm': {
        'id': 'Q12_safety_check',
        'type': 'radio',
        'required': True,
        'labels': {
            'en': 'Please confirm:',
            'hi': 'कृपया पुष्टि करें:',
            'ta': 'தயவுசெய்து உறுதிப்படுத்தவும்:',
            'te': 'దయచేసి నిర్ధారించండి:',
            'ml': 'ദയവായി സ്ഥിരീകരിക്കുക:'
        },
        'options': {
            'en': ['I understand I should contact my doctor if symptoms worsen', 'I have questions for a healthcare professional'],
            'hi': ['मैं समझता हूं कि अगर लक्षण बिगड़ते हैं तो मुझे अपने डॉक्टर से संपर्क करना चाहिए', 'मेरे पास स्वास्थ्य पेशेवर के लिए प्रश्न हैं'],
            'ta': ['அறிகுறிகள் மோசமடைந்தால் என் மருத்துவரை தொடர்பு கொள்ள வேண்டும் என்று புரிந்துகொள்கிறேன்', 'சுகாதார நிபுணரிடம் கேள்விகள் உள்ளன'],
            'te': ['లక్షణాలు మరింత తీవ్రమైతే నేను నా వైద్యుడిని సంప్రదించాలని అర్థం చేసుకున్నాను', 'నాకు ఆరోగ్య నిపుణులకు ప్రశ్నలు ఉన్నాయి'],
            'ml': ['രോഗലക്ഷണങ്ങൾ വഷളായാൽ ഞാൻ എന്റെ ഡോക്ടറെ ബന്ധപ്പെടണമെന്ന് എനിക്ക് മനസ്സിലായി', 'എനിക്ക് ഒരു ആരോഗ്യ വിദഗ്ധനോട് ചോദ്യങ്ങളുണ്ട്']
        },
        'values': ['confirmed', 'has_questions']
    }
}

# Language names for the selector
LANGUAGE_NAMES = {
    'en': 'English',
    'hi': 'हिंदी (Hindi)',
    'ta': 'தமிழ் (Tamil)',
    'te': 'తెలుగు (Telugu)',
    'ml': 'മലയാളം (Malayalam)'
}


# =============================================================================
# FORM TOKEN MANAGEMENT
# =============================================================================

# In-memory storage for form tokens (use database in production)
_form_tokens: Dict[str, Dict[str, Any]] = {}


def generate_form_token(visit_id: int, patient_id: str, form_type: str = 'initial') -> str:
    """
    Generate a unique token for a form submission.
    
    Args:
        visit_id: Visit ID
        patient_id: Patient ID
        form_type: 'initial' or 'clarification'
        
    Returns:
        Unique form token
    """
    token = secrets.token_urlsafe(32)
    _form_tokens[token] = {
        'visit_id': visit_id,
        'patient_id': patient_id,
        'form_type': form_type,
        'created_at': datetime.now().isoformat(),
        'filled': False
    }
    return token


def validate_form_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Validate a form token and return associated data.
    
    Args:
        token: Form token to validate
        
    Returns:
        Token data if valid, None otherwise
    """
    return _form_tokens.get(token)


def mark_form_filled(token: str, responses: Dict[str, Any]) -> bool:
    """
    Mark a form as filled with responses.
    
    Args:
        token: Form token
        responses: Form responses
        
    Returns:
        True if successful
    """
    if token in _form_tokens:
        _form_tokens[token]['filled'] = True
        _form_tokens[token]['filled_at'] = datetime.now().isoformat()
        _form_tokens[token]['responses'] = responses
        return True
    return False


def check_form_completed(visit_id: int) -> bool:
    """
    Check if a form has been completed for a visit.
    
    Args:
        visit_id: Visit ID to check
        
    Returns:
        True if form was filled
    """
    for token_data in _form_tokens.values():
        if token_data.get('visit_id') == visit_id and token_data.get('filled'):
            return True
    return False


def get_form_responses(visit_id: int) -> Optional[Dict[str, Any]]:
    """
    Get form responses for a visit.
    
    Args:
        visit_id: Visit ID
        
    Returns:
        Form responses if available
    """
    for token_data in _form_tokens.values():
        if token_data.get('visit_id') == visit_id and token_data.get('filled'):
            return token_data.get('responses')
    return None


# =============================================================================
# FORM URL GENERATION
# =============================================================================

def generate_form_url(visit_id: int, patient_id: str, language: str = 'en',
                      form_type: str = 'initial') -> str:
    """
    Generate a unique form URL for a patient.
    
    Args:
        visit_id: Visit ID
        patient_id: Patient ID
        language: Preferred language
        form_type: 'initial' or 'clarification'
        
    Returns:
        Complete form URL
    """
    token = generate_form_token(visit_id, patient_id, form_type)
    return f"{FORM_BASE_URL}/{token}?lang={language}"


def generate_clarification_form_url(visit_id: int, patient_id: str,
                                     missing_questions: List[str],
                                     language: str = 'en') -> str:
    """
    Generate a clarification form URL with only missing questions.
    
    Args:
        visit_id: Visit ID
        patient_id: Patient ID
        missing_questions: List of question IDs that need answers
        language: Preferred language
        
    Returns:
        Clarification form URL
    """
    token = generate_form_token(visit_id, patient_id, 'clarification')
    _form_tokens[token]['missing_questions'] = missing_questions
    questions_param = ','.join(missing_questions)
    return f"{FORM_BASE_URL}/clarification/{token}?lang={language}&q={questions_param}"


# =============================================================================
# FORM DATA HELPERS
# =============================================================================

def get_questions_for_language(language: str = 'en') -> Dict[str, Any]:
    """
    Get all form questions in the specified language.
    
    Args:
        language: Language code
        
    Returns:
        Dict of questions with labels in specified language
    """
    result = {}
    for key, question in FORM_QUESTIONS.items():
        q = question.copy()
        q['label'] = q['labels'].get(language, q['labels']['en'])
        if 'options' in q:
            q['option_labels'] = q['options'].get(language, q['options']['en'])
        result[key] = q
    return result


def process_form_submission(token: str, form_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a form submission.
    
    Args:
        token: Form token
        form_data: Submitted form data
        
    Returns:
        Processing result
    """
    token_data = validate_form_token(token)
    if not token_data:
        return {'success': False, 'error': 'Invalid or expired form token'}
    
    if token_data.get('filled'):
        return {'success': False, 'error': 'Form already submitted'}
    
    # Process and validate responses
    responses = {}
    for key, value in form_data.items():
        if key in FORM_QUESTIONS:
            responses[FORM_QUESTIONS[key]['id']] = value
    
    # Mark form as filled
    mark_form_filled(token, responses)
    
    logger.info(f"✅ Form submitted: visit_id={token_data['visit_id']}, responses={len(responses)}")
    
    return {
        'success': True,
        'visit_id': token_data['visit_id'],
        'patient_id': token_data['patient_id'],
        'responses': responses
    }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("📝 Form Service - Test Mode")
    print("=" * 60)
    
    # Test token generation
    url = generate_form_url(1, 'P001', 'en')
    print(f"\n📋 Generated form URL: {url}")
    
    # Test question retrieval
    questions = get_questions_for_language('hi')
    print(f"\n📝 Questions in Hindi ({len(questions)} total):")
    for key, q in list(questions.items())[:3]:
        print(f"  - {q['label'][:50]}...")
    
    # Test form completion check
    completed = check_form_completed(1)
    print(f"\n✅ Form completed for visit 1: {completed}")
    
    print("\n" + "=" * 60)
