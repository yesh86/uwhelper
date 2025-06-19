"""
Terminal Chat Interface for Insurance RAG System
Direct test of the complete RAG pipeline without Flask/browser
"""

import os
import sys
from datetime import datetime

# Set API key first
os.environ['ANTHROPIC_API_KEY'] = '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'  # Replace with your actual key

# Import the RAG system from your app.py
try:
    from app import ClaudeInsuranceRAG

    print("✅ Successfully imported RAG system from app.py")
except ImportError as e:
    print(f"❌ Failed to import RAG system: {e}")
    print("Make sure app.py is in the same directory")
    sys.exit(1)


class TerminalChat:
    def __init__(self):
        """Initialize the terminal chat interface"""
        print("🚀 Initializing Terminal Insurance Chat...")
        print("=" * 60)

        try:
            self.rag_system = ClaudeInsuranceRAG()
            print("✅ RAG system loaded successfully!")
            print(f"📚 Knowledge base: {len(self.rag_system.chunks_metadata)} chunks")
            print(f"🤖 Ready to chat with Claude!")
        except Exception as e:
            print(f"❌ Failed to initialize RAG system: {e}")
            sys.exit(1)

    def format_response(self, result):
        """Format the response for terminal display"""
        print("\n" + "=" * 60)
        print("🤖 CLAUDE'S RESPONSE:")
        print("=" * 60)

        # Main answer
        print(result['answer'])

        # Metadata
        print("\n" + "-" * 40)
        print("📊 RESPONSE METADATA:")
        print(f"   🎯 Confidence: {result['confidence']}%")
        print(f"   ⏱️  Response time: {result['response_time']:.2f}s")
        print(f"   📄 Chunks used: {result['chunks_used']}")
        print(f"   🔍 Query type: {result['query_type']}")

        # Sources
        if result['sources']:
            print("\n📚 SOURCES REFERENCED:")
            for i, source in enumerate(result['sources'], 1):
                print(f"   {i}. {source}")

        print("-" * 40)

    def run(self):
        """Run the interactive chat loop"""
        print("\n" + "=" * 60)
        print("💬 TERMINAL INSURANCE CHAT")
        print("=" * 60)
        print("Ask me anything about insurance policies, coverage, claims, or forms!")
        print("Type 'quit' or 'exit' to end the chat.")
        print("Type 'help' for sample questions.")
        print("=" * 60)

        while True:
            try:
                # Get user input
                print("\n" + ">" * 3, end=" ")
                user_input = input().strip()

                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thanks for using the Insurance Chat! Goodbye!")
                    break

                if user_input.lower() in ['help', 'h']:
                    self.show_help()
                    continue

                if not user_input:
                    print("Please enter a question or type 'help' for examples.")
                    continue

                # Process the query
                print(f"\n🔍 Processing: {user_input}")
                start_time = datetime.now()

                result = self.rag_system.answer_question(user_input)

                # Display the response
                self.format_response(result)

                # Show any errors
                if not result.get('success', True):
                    print(f"\n⚠️ Warning: {result.get('error', 'Unknown error')}")

            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error processing your question: {e}")
                continue

    def show_help(self):
        """Show sample questions"""
        print("\n" + "=" * 50)
        print("💡 SAMPLE QUESTIONS:")
        print("=" * 50)

        samples = [
            "What does equipment breakdown protection cover?",
            "How do I file a claim?",
            "What is BM_00_20_07_01 all about?",
            "What are the exclusions for this coverage?",
            "Explain liability insurance limits",
            "What is workers compensation coverage?",
            "Tell me about commercial auto insurance",
            "What does professional liability cover?",
            "Explain the claims process",
            "What are deductibles and how do they work?"
        ]

        for i, sample in enumerate(samples, 1):
            print(f"   {i:2d}. {sample}")

        print("=" * 50)


def test_single_query():
    """Test a single query without interactive mode"""
    print("🧪 SINGLE QUERY TEST")
    print("=" * 40)

    try:
        rag = ClaudeInsuranceRAG()
        test_query = "What does equipment breakdown protection cover?"

        print(f"Testing query: {test_query}")
        result = rag.answer_question(test_query)

        print(f"\nResponse: {result['answer'][:200]}...")
        print(f"Confidence: {result['confidence']}%")
        print(f"Sources: {len(result['sources'])} documents")
        print(f"Success: {result['success']}")

        return result['success']

    except Exception as e:
        print(f"❌ Single query test failed: {e}")
        return False


def main():
    """Main function"""
    print("🏢 INSURANCE RAG TERMINAL INTERFACE")
    print("=" * 60)

    # Quick API key check
    api_key = os.environ.get('ANTHROPIC_API_KEY', '')
    if not api_key or not api_key.startswith('sk-ant-'):
        print("❌ API key not set correctly!")
        print("Please edit this file and set your API key in line 13")
        return

    print(f"🔑 API key: {api_key[:20]}...")

    # Ask user what they want to do
    print("\nChoose an option:")
    print("1. Interactive chat")
    print("2. Single query test")
    print("3. Exit")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == '1':
        chat = TerminalChat()
        chat.run()
    elif choice == '2':
        success = test_single_query()
        if success:
            print("\n✅ Single query test passed!")
        else:
            print("\n❌ Single query test failed!")
    elif choice == '3':
        print("Goodbye!")
    else:
        print("Invalid choice. Please run again.")


if __name__ == "__main__":
    main()