"""
QUICK REFERENCE: How to Configure Different LLMs for Different Agents

This file shows simple, copy-paste examples for common scenarios.
"""

# ============================================================================
# SCENARIO 1: Change a Single Agent's LLM
# ============================================================================

from agent_llm_config import update_agent_llm_config

# Make NL-to-SQL agent use OpenAI GPT-4 for better SQL generation
update_agent_llm_config(
    "nl_to_sql",
    provider="openai",
    model="gpt-4",
    temperature=0.1
)


# ============================================================================
# SCENARIO 2: View Current Configuration
# ============================================================================

from agent_llm_config import print_agent_llm_config, get_all_agent_llm_configs

# Print formatted configuration
print_agent_llm_config()

# Or get as dictionary
configs = get_all_agent_llm_configs()
for agent, config in configs.items():
    print(f"{agent}: {config['provider']}/{config['model']}")


# ============================================================================
# SCENARIO 3: Setup Different Providers Based on Environment
# ============================================================================

import os
from agent_llm_config import update_agent_llm_config

# Use OpenAI if available
if os.getenv("OPENAI_API_KEY"):
    # Complex tasks use GPT-4
    update_agent_llm_config("nl_to_sql", provider="openai", model="gpt-4")
    update_agent_llm_config("decision_intelligence", provider="openai", model="gpt-4")
else:
    # Fallback to local Ollama
    update_agent_llm_config("nl_to_sql", provider="ollama", model="llama3.2")
    update_agent_llm_config("decision_intelligence", provider="ollama", model="llama3.2")

# Always use Anthropic for creative tasks if available
if os.getenv("ANTHROPIC_API_KEY"):
    update_agent_llm_config("text", provider="anthropic", model="claude-sonnet-4")


# ============================================================================
# SCENARIO 4: Optimize for Cost (Use Cheap Models)
# ============================================================================

from agent_llm_config import update_agent_llm_config

agents = [
    "query_analysis",
    "intent_identifier",
    "nl_to_sql",
    "data_observation",
    "forecasting",
    "decision_intelligence",
    "summarization",
    "text"
]

# Use cheapest available option: local Ollama
for agent in agents:
    update_agent_llm_config(agent, provider="ollama", model="llama3.2")


# ============================================================================
# SCENARIO 5: Optimize for Quality (Use Best Models)
# ============================================================================

from agent_llm_config import update_agent_llm_config

# Classification - use fastest model
update_agent_llm_config("query_analysis", provider="openai", model="gpt-3.5-turbo")

# SQL generation - use best model
update_agent_llm_config("nl_to_sql", provider="openai", model="gpt-4-turbo")

# Analysis - use capable model
update_agent_llm_config("decision_intelligence", provider="anthropic", model="claude-opus")

# Text generation - use balanced model
update_agent_llm_config("text", provider="openai", model="gpt-3.5-turbo")


# ============================================================================
# SCENARIO 6: Mixed Setup (Different Providers for Different Tasks)
# ============================================================================

from agent_llm_config import update_agent_llm_config

# Fast & cheap - use local model
update_agent_llm_config("query_analysis", provider="ollama", model="llama3.2")
update_agent_llm_config("data_observation", provider="ollama", model="llama3.2")
update_agent_llm_config("forecasting", provider="ollama", model="llama3.2")

# Complex - use powerful cloud model
update_agent_llm_config("nl_to_sql", provider="openai", model="gpt-4")

# Creative - use Anthropic
update_agent_llm_config("decision_intelligence", provider="anthropic", model="claude-sonnet-4")
update_agent_llm_config("text", provider="anthropic", model="claude-sonnet-4")

# Formatting - use cheap fast model
update_agent_llm_config("summarization", provider="openai", model="gpt-3.5-turbo")


# ============================================================================
# SCENARIO 7: Use Agent's LLM in Your Code
# ============================================================================

from agent_llm_config import get_agent_llm
from langchain_core.prompts import ChatPromptTemplate

# Get the LLM for an agent
llm = get_agent_llm("nl_to_sql")

# Use it to invoke a prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a SQL expert"),
    ("user", "Generate SQL to fetch all users")
])

response = llm.invoke(prompt.format_messages())
print(response.content)


# ============================================================================
# SCENARIO 8: Temporary LLM Switch for Testing
# ============================================================================

from agent_llm_config import update_agent_llm_config, get_all_agent_llm_configs

# Save original configuration
original_config = get_all_agent_llm_configs()

# Test with different LLM
print("Testing with OpenAI GPT-4...")
update_agent_llm_config("forecasting", provider="openai", model="gpt-4")
# ... run your tests ...

# Restore original
print("Restoring original configuration...")
for agent, config in original_config.items():
    update_agent_llm_config(
        agent,
        provider=config.get("provider"),
        model=config.get("model"),
        temperature=config.get("temperature")
    )


# ============================================================================
# SCENARIO 9: Configure in agent_llm_config.py (Persistent Configuration)
# ============================================================================

"""
Edit agent_llm_config.py directly for permanent changes:

AGENT_LLM_CONFIG = {
    # Fast classification
    "query_analysis": {
        "provider": "ollama",
        "model": "llama3.2",
        "temperature": 0,
    },
    
    # Complex SQL generation - use powerful model
    "nl_to_sql": {
        "provider": "openai",  # ← Changed this
        "model": "gpt-4",       # ← Changed this
        "temperature": 0.1,
    },
    
    "data_observation": {
        "provider": "ollama",
        "model": "llama3.2",
        "temperature": 0,
    },
    
    # ... more agents ...
}
"""


# ============================================================================
# SCENARIO 10: Handle Missing API Keys
# ============================================================================

from agent_llm_config import update_agent_llm_config
import os

def setup_llms():
    """Setup LLMs based on available API keys"""
    
    # Default to local model
    default_provider = "ollama"
    default_model = "llama3.2"
    
    # Override if OpenAI available
    if os.getenv("OPENAI_API_KEY"):
        default_provider = "openai"
        default_model = "gpt-4"
    
    # Override if Anthropic available
    if os.getenv("ANTHROPIC_API_KEY"):
        default_provider = "anthropic"
        default_model = "claude-sonnet-4"
    
    # Apply to all agents
    agents = [
        "query_analysis", "intent_identifier", "nl_to_sql",
        "data_observation", "forecasting", "decision_intelligence",
        "summarization", "text"
    ]
    
    for agent in agents:
        update_agent_llm_config(agent, provider=default_provider, model=default_model)
    
    print(f"Setup complete. Using: {default_provider}/{default_model}")

# Call at startup
setup_llms()


# ============================================================================
# DETAILED AGENT REFERENCE
# ============================================================================

"""
Available agents and suggested optimal configurations:

1. query_analysis
   - Purpose: Fast classification of user intent
   - Suggested: Fast, cheap model (Ollama, GPT-3.5)
   - Temperature: 0 (deterministic)

2. intent_identifier
   - Purpose: Plan execution (minimal logic)
   - Suggested: Fast model
   - Temperature: 0

3. nl_to_sql (MOST IMPORTANT)
   - Purpose: Convert natural language to SQL
   - Suggested: Powerful model (GPT-4, Claude)
   - Temperature: 0.1 (slight creativity for variations)
   - Note: Biggest impact on query quality

4. data_observation
   - Purpose: Process and analyze data
   - Suggested: Balanced model
   - Temperature: 0-0.1

5. forecasting
   - Purpose: Generate forecast data
   - Suggested: Balanced model
   - Temperature: 0.1-0.3

6. decision_intelligence
   - Purpose: Business insights and recommendations
   - Suggested: Capable model (Claude, GPT-4)
   - Temperature: 0.2-0.3 (some creativity)

7. summarization
   - Purpose: Format responses for display
   - Suggested: Fast, cheap model
   - Temperature: 0

8. text
   - Purpose: General conversation
   - Suggested: Capable model, higher temperature
   - Temperature: 0.5-0.7 (more creative)
"""


# ============================================================================
# COMMON MISTAKES & SOLUTIONS
# ============================================================================

"""
MISTAKE 1: Using wrong provider name
❌ update_agent_llm_config("query_analysis", provider="chatgpt")
✅ update_agent_llm_config("query_analysis", provider="openai")

MISTAKE 2: Missing required API key
❌ update_agent_llm_config("query_analysis", provider="openai")  # without OPENAI_API_KEY set
✅ First set: export OPENAI_API_KEY="sk-..."

MISTAKE 3: Typo in agent name
❌ update_agent_llm_config("query_analisys", ...)  # typo!
✅ update_agent_llm_config("query_analysis", ...)

MISTAKE 4: Wrong temperature for task
❌ update_agent_llm_config("nl_to_sql", temperature=0.9)  # too creative!
✅ update_agent_llm_config("nl_to_sql", temperature=0.1)  # precise

Valid agent names:
- query_analysis
- intent_identifier
- nl_to_sql
- data_observation
- forecasting
- decision_intelligence
- summarization
- text
"""
