# llm_factory.py
"""
LLM Factory - Platform-independent LLM provider
Supports: Ollama, OpenAI, Anthropic, Google Gemini, vLLM
"""

from langchain_core.language_models import BaseChatModel
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()


class LLMFactory:
    """Factory class for creating LLM instances based on provider"""
    
    @staticmethod
    def create_llm(
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> BaseChatModel:
        """
        Create an LLM instance based on the specified provider.
        
        Args:
            provider: LLM provider (ollama, openai, anthropic, gemini, vllm)
            model: Model name (overrides env variable)
            temperature: Temperature setting (overrides env variable)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            BaseChatModel: Configured LLM instance
        """
        # Get configuration from environment
        provider = provider or os.getenv("LLM_PROVIDER", "ollama").lower()
        temperature = temperature if temperature is not None else float(os.getenv("TEMPERATURE", "0"))
        
        print(f"[LLM_FACTORY] Initializing {provider.upper()} provider...")
        
        if provider == "ollama":
            return LLMFactory._create_ollama(model, temperature, **kwargs)
        elif provider == "openai":
            return LLMFactory._create_openai(model, temperature, **kwargs)
        elif provider == "anthropic":
            return LLMFactory._create_anthropic(model, temperature, **kwargs)
        elif provider == "gemini":
            return LLMFactory._create_gemini(model, temperature, **kwargs)
        elif provider == "vllm":
            return LLMFactory._create_vllm(model, temperature, **kwargs)
        else:
            raise ValueError(
                f"Unsupported LLM provider: {provider}. "
                f"Supported: ollama, openai, anthropic, gemini, vllm"
            )
    
    @staticmethod
    def _create_ollama(model: Optional[str], temperature: float, **kwargs) -> BaseChatModel:
        """Create Ollama LLM instance"""
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError(
                "langchain-ollama not installed. "
                "Install with: pip install langchain-ollama"
            )
        
        model = model or os.getenv("MODEL_NAME", "llama3.2")
        base_url = kwargs.get("base_url") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        print(f"[LLM_FACTORY] Ollama model: {model}")
        
        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=base_url,
            **{k: v for k, v in kwargs.items() if k != "base_url"}
        )
    
    @staticmethod
    def _create_openai(model: Optional[str], temperature: float, **kwargs) -> BaseChatModel:
        """Create OpenAI LLM instance"""
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "langchain-openai not installed. "
                "Install with: pip install langchain-openai"
            )
        
        model = model or os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
        api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        print(f"[LLM_FACTORY] OpenAI model: {model}")
        
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            **{k: v for k, v in kwargs.items() if k != "api_key"}
        )
    
    @staticmethod
    def _create_anthropic(model: Optional[str], temperature: float, **kwargs) -> BaseChatModel:
        """Create Anthropic (Claude) LLM instance"""
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "langchain-anthropic not installed. "
                "Install with: pip install langchain-anthropic"
            )
        
        model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        api_key = kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        
        print(f"[LLM_FACTORY] Anthropic model: {model}")
        
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=api_key,
            **{k: v for k, v in kwargs.items() if k != "api_key"}
        )
    
    @staticmethod
    def _create_gemini(model: Optional[str], temperature: float, **kwargs) -> BaseChatModel:
        """Create Google Gemini LLM instance"""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                "langchain-google-genai not installed. "
                "Install with: pip install langchain-google-genai"
            )
        
        model = model or os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
        api_key = kwargs.get("api_key") or os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        print(f"[LLM_FACTORY] Gemini model: {model}")
        
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=api_key,
            **{k: v for k, v in kwargs.items() if k != "api_key"}
        )
    
    @staticmethod
    def _create_vllm(model: Optional[str], temperature: float, **kwargs) -> BaseChatModel:
        """Create vLLM LLM instance (OpenAI-compatible API)"""
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "langchain-openai not installed. "
                "Install with: pip install langchain-openai"
            )
        
        model = model or os.getenv("VLLM_MODEL", "meta-llama/Llama-2-7b-chat-hf")
        base_url = kwargs.get("base_url") or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        api_key = kwargs.get("api_key") or os.getenv("VLLM_API_KEY", "EMPTY")
        
        print(f"[LLM_FACTORY] vLLM model: {model}")
        
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key,
            **{k: v for k, v in kwargs.items() if k not in ["base_url", "api_key"]}
        )


def get_llm(provider: Optional[str] = None, **kwargs) -> BaseChatModel:
    """
    Convenience function to get LLM instance.
    
    Args:
        provider: LLM provider (ollama, openai, anthropic, gemini, vllm)
        **kwargs: Additional configuration parameters
        
    Returns:
        BaseChatModel: Configured LLM instance
        
    Example:
        >>> llm = get_llm()  # Uses LLM_PROVIDER from .env
        >>> llm = get_llm(provider="openai")
        >>> llm = get_llm(provider="anthropic", model="claude-opus-4-20241120")
    """
    return LLMFactory.create_llm(provider=provider, **kwargs)