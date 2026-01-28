"""
Agent LLM Configuration
Defines which LLM each agent should use
"""

from llm_factory import get_llm
from langchain_core.language_models import BaseChatModel
from typing import Dict, Optional, Any


# =========================================================================
# AGENT LLM CONFIGURATION - Customize which agent uses which LLM
# =========================================================================

AGENT_LLM_CONFIG = {
    # Query Analysis Agent - fast model for classification
    "query_analysis": {
        "provider": "ollama",  # or "openai", "anthropic", "gemini", "vllm"
        "model": "llama3.2",
        "temperature": 0,
    },
    
    # Intent Identifier Agent
    "intent_identifier": {
        "provider": "ollama",
        "model": "llama3.2",
        "temperature": 0,
    },
    
    # NL to SQL Agent - complex reasoning, use more capable model
    "nl_to_sql": {
        "provider": "ollama",  # Try "openai" for better SQL generation
        "model": "qwen2.5:latest",
        "temperature": 0.0,
    },
    
    # Data Observation Agent
    "data_observation": {
        "provider": "ollama",
        "model": "llama3.2",
        "temperature": 0,
    },
    
    # Forecasting Agent - needs reasoning
    "forecasting": {
        "provider": "ollama",
        "model": "llama3.2",
        "temperature": 0.2,
    },
    
    # Decision Intelligence Agent - needs reasoning and insights
    "decision_intelligence": {
        "provider": "ollama",
        "model": "llama3.2",
        "temperature": 0.3,
    },
    
    # Summarization Agent
    "summarization": {
        "provider": "ollama",
        "model": "llama3.2",
        "temperature": 0,
    },
    
    # Text Agent - general conversation
    "text": {
        "provider": "ollama",
        "model": "llama3.2",
        "temperature": 0.7,
    },
}


# Cache for LLM instances to avoid recreating them
_llm_cache: Dict[str, BaseChatModel] = {}


def get_agent_llm(agent_name: str) -> BaseChatModel:
    """
    Get the LLM instance configured for a specific agent.
    
    Args:
        agent_name: Name of the agent (e.g., "query_analysis", "nl_to_sql")
        
    Returns:
        BaseChatModel: Configured LLM instance for the agent
        
    Example:
        >>> llm = get_agent_llm("nl_to_sql")
        >>> response = llm.invoke(messages)
    """
    # Check cache first
    if agent_name in _llm_cache:
        cached_config = AGENT_LLM_CONFIG.get(agent_name, {})
        print(f"[AGENT_LLM] Using CACHED LLM for '{agent_name}': "
              f"{cached_config.get('provider')}/{cached_config.get('model')}")
        return _llm_cache[agent_name]
    
    # Get config for agent
    if agent_name not in AGENT_LLM_CONFIG:
        raise ValueError(
            f"No LLM configuration found for agent '{agent_name}'. "
            f"Available agents: {list(AGENT_LLM_CONFIG.keys())}"
        )
    
    config = AGENT_LLM_CONFIG[agent_name]
    
    print(f"[AGENT_LLM] Creating NEW LLM for '{agent_name}': "
          f"{config.get('provider')}/{config.get('model')} (temp={config.get('temperature')})")
    
    # Create LLM instance
    llm = get_llm(
        provider=config.get("provider"),
        model=config.get("model"),
        temperature=config.get("temperature"),
    )
    
    # Cache it
    _llm_cache[agent_name] = llm
    
    print(f"[AGENT_LLM] ✓ Loaded LLM for '{agent_name}': "
          f"{config.get('provider')}/{config.get('model')}")
    
    return llm


def update_agent_llm_config(
    agent_name: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    **kwargs
) -> None:
    """
    Update LLM configuration for a specific agent at runtime.
    
    Args:
        agent_name: Name of the agent
        provider: New provider (e.g., "openai", "anthropic")
        model: New model name
        temperature: New temperature value
        **kwargs: Additional configuration options
        
    Example:
        >>> update_agent_llm_config(
        ...     "nl_to_sql",
        ...     provider="openai",
        ...     model="gpt-4",
        ...     temperature=0.1
        ... )
    """
    if agent_name not in AGENT_LLM_CONFIG:
        raise ValueError(f"Unknown agent: {agent_name}")
    
    config = AGENT_LLM_CONFIG[agent_name]
    
    # Update config
    if provider is not None:
        config["provider"] = provider
    if model is not None:
        config["model"] = model
    if temperature is not None:
        config["temperature"] = temperature
    config.update(kwargs)
    
    # Clear cache for this agent
    if agent_name in _llm_cache:
        del _llm_cache[agent_name]
    
    print(f"[AGENT_LLM] Updated config for '{agent_name}': {config}")


def get_all_agent_llm_configs() -> Dict[str, Dict[str, Any]]:
    """
    Get all agent LLM configurations.
    
    Returns:
        Dictionary of all agent configurations
        
    Example:
        >>> configs = get_all_agent_llm_configs()
        >>> for agent, config in configs.items():
        ...     print(f"{agent}: {config}")
    """
    return AGENT_LLM_CONFIG.copy()


def clear_agent_llm_cache(agent_name: Optional[str] = None) -> None:
    """
    Clear LLM cache for specific agent or all agents.
    
    Useful when you change configuration and want fresh LLM instances.
    
    Args:
        agent_name: Specific agent to clear, or None to clear all
        
    Example:
        >>> clear_agent_llm_cache("nl_to_sql")  # Clear specific agent
        >>> clear_agent_llm_cache()  # Clear all agents
    """
    global _llm_cache
    
    if agent_name is None:
        count = len(_llm_cache)
        _llm_cache.clear()
        print(f"[AGENT_LLM] Cleared cache for ALL {count} agents")
    else:
        if agent_name in _llm_cache:
            del _llm_cache[agent_name]
            print(f"[AGENT_LLM] Cleared cache for '{agent_name}'")
        else:
            print(f"[AGENT_LLM] No cache entry for '{agent_name}'")


def print_agent_llm_config() -> None:
    """Print current agent LLM configurations in a formatted way."""
    print("\n" + "="*70)
    print("AGENT LLM CONFIGURATION")
    print("="*70)
    for agent_name, config in AGENT_LLM_CONFIG.items():
        cached = "✓ CACHED" if agent_name in _llm_cache else "  fresh"
        print(f"\n{agent_name.upper():30} {cached}")
        print(f"  Provider:    {config.get('provider')}")
        print(f"  Model:       {config.get('model')}")
        print(f"  Temperature: {config.get('temperature')}")
    print("\n" + "="*70 + "\n")
