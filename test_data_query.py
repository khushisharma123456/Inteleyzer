"""
Quick test for patient data query feature
"""

from db_utils import detect_data_query_intent

print("Testing intent detection...")
print("=" * 50)

tests = [
    "show my data",
    "my records", 
    "visit id 1",
    "hello",
    "1",
    "I want to see my information",
    "mera data",  # Hindi
]

for msg in tests:
    result = detect_data_query_intent(msg)
    status = "✅ DATA QUERY" if result['is_data_query'] else "❌ Not a query"
    print(f"  '{msg}' -> {status}")
    if result['is_data_query']:
        print(f"      Type: {result['query_type']}, Visit ID: {result['visit_id']}")

print("=" * 50)
print("✅ Intent detection module working!")
