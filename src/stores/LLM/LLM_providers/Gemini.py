from ..LLMinterface import LLMInterface
from google import genai
from google.genai import types
import logging
from ..LLMEnums import GeminiEnums
from typing import List , Union

class Gemini_provider(LLMInterface):
    def __init__(self, api_key: str, default_input_max_tokens: int = 1000, default_output_max_tokens: int = 1000, default_temperature: float = 0.1):
        
        self.api_key = api_key
        self.default_input_max_tokens = default_input_max_tokens
        self.default_output_max_tokens = default_output_max_tokens
        self.default_temperature = default_temperature

        self.generator_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None

        self.client = genai.Client(api_key=self.api_key)

        self.enums = GeminiEnums

        self.logger = logging.getLogger(__name__)

    def set_generator(self, model_id: str):
        self.generator_model_id = model_id
        self.logger.info(f"Generator model set to: {model_id}")

    def set_embedding(self, model_id: str, embedding_size: int = 1536):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size
        self.logger.info(f"Embedding model set to: {model_id}")

    def process_text(self, text: str):
        return text[:self.default_input_max_tokens].strip()
    
    
    def generate_text(self, prompt: str, chat_history: list = None, max_output_tokens: int = 512, temperature: float = 0.1, top_p: float = 0.1):
        if not self.client:
            self.logger.error("Gemini client is not initialized.")
            return None
        
        if not self.generator_model_id:
            self.logger.error("Generator model ID is not set.")
            return None
        
        if chat_history is None:
            chat_history = []
        
        max_output_tokens = max_output_tokens if max_output_tokens is not None else self.default_output_max_tokens
        temperature = temperature if temperature is not None else self.default_temperature
        top_p = top_p if top_p is not None else self.default_top_p

        chat_history.append(self.construct_prompt(prompt=prompt, role="user"))

        try:
            response = self.client.models.generate_content(
                model=self.generator_model_id,
                contents=chat_history,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
            )
            
            if not response or not response.text:
                self.logger.error("No text response returned from Gemini.")
                return None
                
            return response.text

        except Exception as e:
            self.logger.error(f"Error during Gemini generation: {e}")
            return None

    def generate_embedding(self, texts: Union[str, List[str]], document_type: str = None):
        #current gemini embedding model doesn't support batch embedding, so we will loop through the list of texts and get embeddings one by one
        if not self.client:
            self.logger.error("Gemini client was not set")
            return None

        is_single = False

        if isinstance(texts, str):
            is_single = True
            texts = [texts]

        if not self.embedding_model_id:
            self.logger.error("Embedding model for Gemini was not set")
            return None

        if document_type:
            config = types.EmbedContentConfig(
            task_type=document_type,
            output_dimensionality=self.embedding_size,
            )
        else:
            config = None

        vectors = []  
        for t in texts:
            result = self.client.models.embed_content(   
                model=self.embedding_model_id,
                contents=t,
                config=config,
            )
            if result and result.embeddings:
                vectors.append([float(v) for v in result.embeddings[0].values])
            else:
                self.logger.error(f"Embedding failed for text: {t[:50]}...")
                vectors.append(None)

        return vectors[0] if is_single else vectors

    
    def construct_prompt(self, prompt: str, role: str):
        return types.Content(
            role=role,
            parts=[types.Part.from_text(text=prompt)]
        )
