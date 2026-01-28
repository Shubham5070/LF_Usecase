#!/usr/bin/env python
"""
Test script to verify agents are using different LLMs
Run this to check configuration and test each agent's LLM
"""

from agent_llm_config import (
    get_agent_llm,
    get_all_agent_llm_configs,
    print_agent_llm_config,
    clear_agent_llm_cache,
    update_agent_llm_config,
    AGENT_LLM_CONFIG
)

print("\n" + "="*80)
print("AGENT LLM CONFIGURATION TESTER")
print("="*80)

# 1. Show all configurations
print("\n1. CURRENT CONFIGURATION:")
print_agent_llm_config()

# 2. Get each agent's LLM and test
print("\n2. TESTING EACH AGENT'S LLM:")
print("-" * 80)
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

for agent_name in agents:
    config = AGENT_LLM_CONFIG[agent_name]
    print(f"\n{agent_name.upper()}")
    print(f"  Config: {config['provider']}/{config['model']} (temp={config['temperature']})")
    
    try:
        llm = get_agent_llm(agent_name)
        print(f"  Status: ✓ LLM loaded successfully")
        print(f"  Type:   {type(llm).__name__}")
    except Exception as e:
        print(f"  Status: ✗ ERROR - {e}")

# 3. Show cache status
print("\n\n3. CACHE STATUS:")
print_agent_llm_config()

# 4. Test cache clearing
print("\n4. TESTING CACHE CLEAR:")
print("-" * 80)
clear_agent_llm_cache("nl_to_sql")
print("Getting nl_to_sql again (should show NEW instead of CACHED):")
llm = get_agent_llm("nl_to_sql")

# 5. Show final status
print("\n\n5. FINAL CONFIGURATION STATUS:")
print_agent_llm_config()

print("\n" + "="*80)
print("✓ Test Complete!")
print("="*80)
print("\nNOTE: Each time you see '[AGENT_LLM] Creating NEW LLM' in the logs,")
print("it means a fresh LLM instance is being created with the configured model.")
print("\nTo change agent models, edit AGENT_LLM_CONFIG in agent_llm_config.py")
print("or use: update_agent_llm_config('agent_name', model='new_model')")
print("="*80 + "\n")
