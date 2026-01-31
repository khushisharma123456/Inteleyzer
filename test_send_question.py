from dotenv import load_dotenv
load_dotenv()
from app import app, db
from models import Patient, AgentFollowupTracking
from pv_backend.services.whatsapp_chatbot import WhatsAppChatbot, ToneManager

app.app_context().push()

p = Patient.query.filter(Patient.name.like('%Priya%')).first()
t = AgentFollowupTracking.query.filter_by(patient_id=p.id, status='active').first()

print(f'Patient: {p.name}')
print(f'Phone: {p.phone}')
print(f'State: {t.chatbot_state}')
print(f'Index: {t.current_question_index}')

uq = t.unanswered_questions or []
print(f'Total Questions: {len(uq)}')

for i, q in enumerate(uq):
    print(f'  Q{i}: {q.get("question", "NO QUESTION")[:80]}...')

# Send the current question manually
if t.current_question_index < len(uq):
    current_q = uq[t.current_question_index]
    question_text = current_q.get('question', '')
    print(f'\n=== Sending Question {t.current_question_index} ===')
    print(f'Q: {question_text}')
    
    chatbot = WhatsAppChatbot()
    msg = ToneManager.get_message('next_question', 'English', question=question_text)
    print(f'Message: {msg}')
    
    result = chatbot.send_message(p.phone, msg)
    print(f'Send result: {result}')
else:
    print('No more questions to send')
