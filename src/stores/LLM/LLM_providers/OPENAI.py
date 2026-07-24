from ..LLMinterface import LLMInterface
from openai import OpenAI
import logging
from ..LLMEnums import OpenAIEnums
from typing import List , Union

class OPENAI_provider(LLMInterface):
    def __init__(self,api_key:str , api_url : str = None,
                default_input_max_tokens:int = 1000, default_output_max_tokens:int = 1000,
                default_temperature:float = 0.1, default_top_p:float = None):
        
        self.api_key = api_key
        self.api_url = api_url
        self.default_input_max_tokens = default_input_max_tokens
        self.default_output_max_tokens = default_output_max_tokens
        self.default_temperature = default_temperature
        self.default_top_p = default_top_p

        self.generator_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None

        self.client = OpenAI(api_key=self.api_key, api_url=self.api_url)
        self.enums = OpenAIEnums
        self.logger = logging.getLogger(__name__)

    def set_generator(self, model_id:str):
        self.generator_model_id = model_id
        self.logger.info(f"Generator model set to: {model_id}")

    def set_embedding(self, model_id:str, embedding_size:int = 1536):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size
        self.logger.info(f"Embedding model set to: {model_id}")

    def process_text(self, text:str):
        return text[:self.default_input_max_tokens].strip()

    def generate_text(self, prompt:str, chat_history:list = [], max_output_tokens:int = 512, temperature:float = 0.1, top_p:float = None):
        if not self.client:
            self.logger.error("OpenAI client is not initialized.")
            return None
        
        if not self.generator_model_id:
            self.logger.error("Generator model ID is not set.")
            return None
        
        max_output_tokens = max_output_tokens if max_output_tokens is not None else self.default_output_max_tokens
        temperature = temperature if temperature is not None else self.default_temperature
        top_p = top_p if top_p is not None else self.default_top_p

        chat_history.append(self.construct_prompt( prompt = prompt, role = OpenAIEnums.USER.value))

        response = self.client.chat.completions.create(
            model=self.generator_model_id,
            messages=chat_history,
            max_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p
        )

        if not response or not response.choices or len(response.choices) == 0 or not response.choices[0].message:
            self.logger.error("No response returned from OpenAI.")
            return None
        return response.choices[0].message['content']

    def generate_embedding(self, text:Union[str, List[str]] , document_type:str = None):
        if isinstance(text, str):
            text = [text]

        if not self.client:
            self.logger.error("OpenAI client is not initialized.")
            return None
        
        if not self.embedding_model_id:
            self.logger.error("Embedding model ID is not set.")
            return None
        
        response = self.client.embeddings.create(
            model=self.embedding_model_id,
            input=text,
            document_type=document_type
        )
        if not response or not response.data or len(response.data) == 0 or not response.data[0].embedding:
            self.logger.error("No embedding data returned from OpenAI.")
            return None
        
        return [data.embedding for data in response.data]


    def construct_prompt(self, prompt:str, role:str):
        return {
            "role": role,
            "content": prompt
        }
    
    