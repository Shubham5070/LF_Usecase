# llm.py
"""
Backward-compatible LLM module
Uses the new LLM factory under the hood
"""

from llm_factory import get_llm as factory_get_llm
from langchain_core.language_models import BaseChatModel
from typing import Optional


def get_llm(provider: Optional[str] = None, **kwargs) -> BaseChatModel:
    """
    Get LLM instance with platform-independent configuration.
    
    This function is backward compatible and uses the LLM factory.
    
    Args:
        provider: LLM provider (ollama, openai, anthropic, gemini, vllm)
        **kwargs: Additional configuration parameters
        
    Returns:
        BaseChatModel: Configured LLM instance
    """
    return factory_get_llm(provider=provider, **kwargs)