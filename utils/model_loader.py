import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from pydantic import Field, SecretStr

from utils.config_loader import load_config
from utils.settings_loader import settings


class ModelLoader:
    """
    A utility class to load embedding models and LLM models.
    """
    def __init__(self):
        self.settings = settings
        self.config=load_config()
        self._validate_env()
        

    def _validate_env(self):
        """
        Validate necessary environment variables.
        """
        required = ["GOOGLE_API_KEY", "GROQ_API_KEY"]
        missing = []
        for var in required:
            val = getattr(self.settings, var, None)
            # SecretStr requires getting the actual string:
            if isinstance(val, SecretStr):
                val = val.get_secret_value()
            if not val:
                missing.append(var)
        if missing:
            raise EnvironmentError(f"Missing environment variables: {missing}")
        

    def load_embeddings(self):
        """Load embedding model based on config."""
        embedding_cfg = self.config["embedding_model"]
        provider = embedding_cfg.get("default_provider", "google")
        model_name = embedding_cfg["providers"][provider]["model_name"]

        if provider == "google":
            print("---------------------- load_embeddings.google ----------------------")
            os.environ["GOOGLE_API_KEY"] = self.settings.GOOGLE_API_KEY
            return GoogleGenerativeAIEmbeddings(
                model=model_name,
                api_key=self.settings.GOOGLE_API_KEY
            )
        else:
            raise ValueError(f"Embedding provider '{provider}' not supported")
    
    
    def load_llm(self, streaming: bool = False):
        """Load LLM based on config."""
        llm_cfg = self.config["llm"]
        provider = llm_cfg.get("default_provider", "google")
        model_name = llm_cfg["providers"][provider]["model_name"]

        if provider == "google":
            print("---------------------- load_llm.google ----------------------")
            os.environ["GOOGLE_API_KEY"] = self.settings.GOOGLE_API_KEY
            return ChatGoogleGenerativeAI(
                model=model_name,
                api_key=self.settings.GOOGLE_API_KEY,
            )

        elif provider == "groq":
            print("---------------------- load_llm.groq ----------------------")
            os.environ["GROQ_API_KEY"] = self.settings.GROQ_API_KEY
            return ChatGroq(
                model=model_name,
                api_key=self.settings.GROQ_API_KEY,
            )

        else:
            raise ValueError(f"LLM provider '{provider}' not supported")
        
        
    def load_safeguard (self):
        """Load the safeguard llm."""
        
        model_name = self.config["safeguard"]["groq"]["model_name"]
        return ChatGroq(
            model=model_name,
            api_key=self.settings.GROQ_API_KEY,
        )
        



# if __name__ == "__main__":
#     model_loader = ModelLoader()
#     model_loader.load_safeguard()