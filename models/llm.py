# models/llm.py
import os
from config.config import CONFIG

def get_llm():
    """
    Returns a lightweight LLM client descriptor.
    For now this returns either:
      - dict with 'provider':'openai', 'client': openai module, 'model': model_name
      - an instance of ChatGroq (if provider == 'groq')
    The real invocation logic will be in app.py / utils where we call the model.
    """
    provider = CONFIG.get("LLM_PROVIDER", "openai").lower()

    if provider == "openai":
        try:
            import openai
        except Exception as e:
            raise ImportError(
                "openai package not installed. Install via `pip install openai`."
            ) from e

        api_key = CONFIG.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not set. Set it in environment or config/.env"
            )
        openai.api_key = api_key
        return {"provider": "openai", "client": openai, "model": CONFIG.get("OPENAI_MODEL")}

    elif provider == "groq":
        try:
            from langchain_groq import ChatGroq
        except Exception as e:
            raise ImportError(
                "langchain_groq not installed. Install via `pip install langchain-groq`."
            ) from e

        api_key = CONFIG.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not set.")

        groq_model = ChatGroq(api_key=api_key, model=CONFIG.get("GROQ_MODEL"))
        return {"provider": "groq", "client": groq_model, "model": CONFIG.get("GROQ_MODEL")}

    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
