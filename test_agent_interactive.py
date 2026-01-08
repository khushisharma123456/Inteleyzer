"""
🧪 Interactive Test for Conversational Agent

Run this to test the pharmacovigilance follow-up conversation
in your terminal.
"""

from ConversationalAgent import ConversationalAgent

def main():
    print("=" * 60)
    print("🏥 Pharmacovigilance Follow-up Agent - Interactive Test")
    print("=" * 60)
    print("\nType your responses (1, 2, 3, etc.) or 'quit' to exit.\n")
    
    # Create agent
    agent = ConversationalAgent()
    
    # Get test parameters
    visit_id = 1
    patient_id = "TEST_PATIENT_001"
    
    # Ask if revisit
    revisit_input = input("Is this a revisit? (y/n): ").strip().lower()
    is_revisit = revisit_input == 'y'
    
    # Start conversation
    result = agent.start_conversation(
        visit_id=visit_id,
        patient_id=patient_id,
        phone_number="+919876543210",
        is_revisit=is_revisit
    )
    
    print("\n" + "-" * 60)
    print("🤖 AGENT:", result['next_question'])
    print("-" * 60)
    
    # Interactive loop
    while not result.get('conversation_complete', False):
        print()
        user_input = input("👤 YOU: ").strip()
        
        if user_input.lower() == 'quit':
            print("\n👋 Conversation ended by user.")
            break
        
        if not user_input:
            print("⚠️ Please enter a response (1, 2, 3, etc.)")
            continue
        
        # Process input
        result = agent.process_input(visit_id=visit_id, user_input=user_input)
        
        print("\n" + "-" * 60)
        print(f"🤖 AGENT: {result['next_question']}")
        print("-" * 60)
        
        # Show status
        if result.get('safety_flag'):
            print("⚠️ SAFETY FLAG RAISED!")
        
        if result.get('conversation_complete'):
            print("\n✅ Conversation completed successfully!")
    
    # Final status
    print("\n" + "=" * 60)
    print("📊 Final Status:")
    status = agent.get_status(visit_id)
    print(f"   State: {status.get('current_state')}")
    print(f"   Complete: {status.get('conversation_complete')}")
    print(f"   Safety Flag: {status.get('safety_flag')}")
    print(f"   Language: {status.get('preferred_language')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
