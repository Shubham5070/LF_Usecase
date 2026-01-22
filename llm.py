from langchain_ollama import ChatOllama

MODEL_NAME = "llama3.2"
TEMPERATURE = 0


def get_llm():
    return ChatOllama(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
    )
