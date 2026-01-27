# test_llm_providers.py
"""
Test script to verify all LLM providers work correctly
"""

from llm_factory import get_llm
from langchain_core.prompts import ChatPromptTemplate


def test_provider(provider: str, test_query: str = "What is 2+2?"):
    """Test a specific LLM provider"""
    print(f"\n{'='*60}")
    print(f"Testing {provider.upper()} Provider")
    print(f"{'='*60}")
    
    try:
        llm = get_llm(provider=provider)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Answer concisely."),
            ("user", "{query}")
        ])
        
        response = llm.invoke(prompt.format_messages(query=test_query))
        
        print(f"✅ SUCCESS")
        print(f"Response: {response.content[:200]}...")
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")


def main():
    """Test all available providers"""
    test_query = "What is the capital of France? Answer in one word."
    
    # Test each provider
    providers = ["ollama", "openai", "anthropic", "gemini", "vllm"]
    
    for provider in providers:
        try:
            test_provider(provider, test_query)
        except Exception as e:
            print(f"\n❌ {provider.upper()} test failed: {e}")
    
    print(f"\n{'='*60}")
    print("Testing Complete")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()