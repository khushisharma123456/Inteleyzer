from dotenv import load_dotenv
load_dotenv()
from app import app, db
from models import Patient, AgentFollowupTracking
from pv_backend.services.whatsapp_chatbot import WhatsAppChatbot

app.app_context().push()

p = Patient.query.filter(Patient.name.like('%Priya%')).first()
t = AgentFollowupTracking.query.filter_by(patient_id=p.id, status='active').first()

print(f'State: {t.chatbot_state}')
print(f'Index: {t.current_question_index}')
uq = t.unanswered_questions or []
print(f'Unanswered count: {len(uq)}')
if uq:
    print(f'Current Q: {uq[t.current_question_index] if t.current_question_index < len(uq) else "None"}')

# Simulate processing a message
chatbot = WhatsAppChatbot()
result = chatbot.process_incoming_message(t, p, 'test answer')

print(f'Result action: {result.get("action")}')
print(f'Response message: {result.get("response_message", "NO RESPONSE")[:300]}')

# Actually send the message if there is one
if result.get('response_message'):
    send_result = chatbot.send_message(p.phone, result['response_message'])
    print(f'Send result: {send_result}')
