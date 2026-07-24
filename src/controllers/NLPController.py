from .BaseController import BaseController
from models.db_schemas import Project , DataChunk 
from stores.LLM.LLMEnums import DocumentTypeEnums
from typing import List
import logging



class NLPController(BaseController):

    def __init__(self, vector_db_client, generation_client, embedding_client, template_parser):
        super().__init__()
        self.vector_db_client = vector_db_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser

        self.logger = logging.getLogger('uvicorn')

    def create_collection_name(self, project_id: str):
        return f"collection_{self.vector_db_client.default_vector_size}_{project_id}"
    
    async def reset_vector_db_collection(self, project : Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        return await self.vector_db_client.delete_collection(collection_name = collection_name)
    
    async def get_vector_db_collection(self, project : Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        collection_info = await self.vector_db_client.get_collection_info(collection_name = collection_name)
        return collection_info
    
    async def index_into_vector_db(self, project : Project, data_chunks : List[DataChunk] , do_reset : bool = False , chunks_ids : List[int] = None):

        collection_name = self.create_collection_name(project_id=project.project_id)

        texts = [chunk.chunk_text for chunk in data_chunks]
        metadatas = [chunk.chunk_metadata for chunk in data_chunks]

        vectors =  self.embedding_client.generate_embedding(texts = texts , document_type = DocumentTypeEnums.RETRIEVAL_DOCUMENT.value)

        print(f"Indexing {len(vectors)} vectors into collection: {collection_name}")

        await self.vector_db_client.create_collection(collection_name=collection_name, embedding_size=self.embedding_client.embedding_size, do_reset=do_reset)

        await self.vector_db_client.insert_collection_batch(collection_name=collection_name, vectors=vectors, texts=texts, metadatas=metadatas, record_ids=chunks_ids)

        return True
        

    async def search_vector_db(self, project : Project, query_text : str, limit : int = 10):
        collection_name = self.create_collection_name(project_id=project.project_id)

        embedding_vector = self.embedding_client.generate_embedding(
            texts=query_text,
            document_type=DocumentTypeEnums.RETRIEVAL_QUERY.value
        )

        if embedding_vector is None:
            self.logger.error("Failed to generate embedding for the query text.")
            return False

        search_results = await self.vector_db_client.search_collection_by_vector(
            collection_name=collection_name,
            vector=embedding_vector,
            limit=limit
        )

        return search_results

    async def generate_response(self, project : Project , query : str , limit : int = 10):
        answer , full_prompt , chat_history = None , None , None
        search_results = await self.search_vector_db(project=project, query_text=query, limit=limit)

        if not search_results or len(search_results) == 0:
            self.logger.error("No search results found.")
            return answer , full_prompt , chat_history

        system_prompt_template = self.template_parser.get(group="rag", key="system_prompt")
        system_prompt = system_prompt_template.substitute()

        document_prompt = "/n".join([
            self.template_parser.get(group="rag", key="document_template", vars={"index": idx, "content": self.generation_client.process_text(result.text), "score": result.score})
            for idx, result in enumerate(search_results)
        ])

        footer_prompt = self.template_parser.get(group="rag", key="footer_template", vars={"question": query})

        chat_history = [self.generation_client.construct_prompt(prompt=system_prompt, role=self.generation_client.enums.SYSTEM.value)]

        full_prompt = "/n/n".join([
            document_prompt,
            footer_prompt
        ])

        answer = self.generation_client.generate_text(prompt=full_prompt, chat_history=chat_history)

        return answer , full_prompt , chat_history



