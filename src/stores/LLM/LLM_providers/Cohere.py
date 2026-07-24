from ..LLMinterface import LLMInterface
import logging
from ..LLMEnums import CohereEnums , DocumentTypeEnums
from cohere import Client as CohereClient
from typing import List , Union

class Cohere_provider(LLMInterface):
    def __init__(self, api_key:str, api_url:str = None,
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

        self.client = CohereClient(api_key=self.api_key)

        self.enums = CohereEnums
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
    
    def generate_text(self, prompt:str, chat_history:list = [], max_output_tokens:int = None, temperature:float = None, top_p:float = None):
        if not self.client:
            self.logger.error("Cohere client is not initialized.")
            return None
        
        if not self.generator_model_id:
            self.logger.error("Generator model ID is not set.")
            return None
        
        max_output_tokens = max_output_tokens if max_output_tokens is not None else self.default_output_max_tokens
        temperature = temperature if temperature is not None else self.default_temperature
        top_p = top_p if top_p is not None else self.default_top_p

        response = self.client.chat(
            model=self.generator_model_id,
            chat_history=chat_history,
            message=self.process_text(prompt),
            max_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p
        )

        if not response or not response.text:
            self.logger.error("No response received from Cohere API.")
            return None

        return response.text
    
    def generate_embedding(self, text:Union[str, List[str]], document_type:str = DocumentTypeEnums.DOCUMENT.value):
        if isinstance(text, str):
            text = [text]

        if not self.client:
            self.logger.error("Cohere client is not initialized.")
            return None
        
        if not self.embedding_model_id:
            self.logger.error("Embedding model ID is not set.")
            return None
        
        input_type = CohereEnums.QUERY.value if document_type == DocumentTypeEnums.QUERY.value else CohereEnums.DOCUMENT.value

        response = self.client.embed(
            model=self.embedding_model_id,
            texts=[self.process_text(t) for t in text],
            input_type = input_type,
            embedding_types = ['float']
        )

        if not response or not response.embeddings or not response.embeddings.float:
            self.logger.error("No embeddings received from Cohere API.")
            return None

        return [response.embeddings.float for response in response.embeddings]


    def construct_prompt(self, prompt:str, role:str):
        return {"role": role, "content": prompt}