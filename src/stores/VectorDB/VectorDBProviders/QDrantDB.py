from qdrant_client import models , QdrantClient
from ..VectorDBEnums import DistanceMetric
from ..VectorDBInterface import VectorDBInterface
import logging
from typing import List
import json
from models.db_schemas import retrievedchunk

class QDrantDB(VectorDBInterface):
    def __init__(self, db_client : str , distance_metric : str , default_vector_size : int = 786, index_threshold : int = 100):

        self.client = None
        self.db_client = db_client
        self.distance_metric = None
        self.default_vector_size = default_vector_size
        self.index_threshold = index_threshold

        if distance_metric.lower() == DistanceMetric.COSINE.value:
            self.distance_metric = models.Distance.COSINE
        elif distance_metric.lower() == DistanceMetric.EUCLIDEAN.value:
            self.distance_metric = models.Distance.EUCLIDEAN
        elif distance_metric.lower() == DistanceMetric.DOT.value:
            self.distance_metric = models.Distance.DOT
        else:
            raise ValueError(f"Invalid distance metric: {distance_metric}. Supported metrics are: {', '.join([metric.value for metric in DistanceMetric])}")

        self.logger = logging.getLogger('uvicorn')

    async def connect(self):
        self.client = QdrantClient(path=self.db_client)

    async def disconnect(self):
        if self.client:
            self.client = None


    async def is_collection_exists(self, collection_name:str) -> bool:
        return self.client.collection_exists(collection_name=collection_name)
    
    
    async def list_all_collections(self, collection_name: str) -> List:
        return self.client.get_collections(collection_name=collection_name)
    
    async def get_collection_info(self, collection_name:str):
        if await self.is_collection_exists(collection_name):
            collection = self.client.get_collection(collection_name=collection_name)
            collection = json.dumps(collection, default= lambda o: o.__dict__)
            return json.loads(collection)
        else:
            self.logger.warning(f"Collection '{collection_name}' does not exist.")
            return None
    
    
    async def delete_collection(self, collection_name:str):
        if await self.is_collection_exists(collection_name):
            return await self.client.delete_collection(collection_name=collection_name)
        else:
            self.logger.warning(f"Collection '{collection_name}' does not exist. Cannot delete.")


    async def create_collection(self, collection_name:str, embedding_size:int, do_reset:bool = False):
        if do_reset and await self.is_collection_exists(collection_name):
            _ = await self.delete_collection(collection_name)
        if not await self.is_collection_exists(collection_name):
            self.logger.info(f"Creating collection '{collection_name}' with embedding size {embedding_size} and distance metric {self.distance_metric}.")
            _ = self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=embedding_size,
                        distance=self.distance_metric
                    )
                )
            return True
        else:
            self.logger.warning(f"Collection '{collection_name}' already exists. Skipping creation.")
            return False
        
    async def insert_collection(self, collection_name:str, text:str, vector:list, metadata:dict = None, record_id:str = None):
        
        if not await self.is_collection_exists(collection_name):
            self.logger.error(f"Collection '{collection_name}' does not exist. Please create the collection before inserting data.")
            return False
        
        try:
            _ = await self.client.upload_points(
                collection_name=collection_name,
                records=[
                    models.Record(
                        id=record_id,
                        vector=vector,
                        payload={"text": text, "metadata": metadata} if metadata else {"text": text},
                    )
                ]
            )

        except Exception as e:
            self.logger.error(f"Failed to insert record into collection '{collection_name}': {e}")
            return False
        
        return True
    

    async def insert_collection_batch(self, collection_name:str, texts:list, vectors:list, metadatas:list = None, record_ids:list = None, batch_size:int = 100):
        if not await self.is_collection_exists(collection_name):
            self.logger.error(f"Collection '{collection_name}' does not exist. Please create the collection before inserting data.")
            return False
        
        if metadatas is None:
            metadatas = [None] * len(texts)

        try:
            for i in range(0, len(texts), batch_size):
                batch_end = min(i + batch_size, len(texts))
                batch_texts = texts[i: batch_end]
                batch_vectors = vectors[i: batch_end]
                batch_metadatas = metadatas[i: batch_end]
                batch_record_ids = record_ids[i: batch_end]

                records = [
                    models.Record(
                        id=record_id,
                        vector=vector,
                        payload={"text": text, "metadata": metadata} if metadata else {"text": text},
                    )
                    for text, vector, metadata, record_id in zip(batch_texts, batch_vectors, batch_metadatas, batch_record_ids)
                ]

                _ = self.client.upload_points(
                    collection_name=collection_name,
                    points=records
                )

                return True
        except Exception as e:
            self.logger.error(f"Failed to insert batch records into collection '{collection_name}': {e}")
            return False
        
    async def search_collection_by_vector(self, collection_name:str , vector:list , limit:int = 10):
        result = self.client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=limit
        )
        if result is None or not result:
            self.logger.warning(f"No results found for the given vector in collection '{collection_name}'.")
            return []
        
        return [
            retrievedchunk(
                **{
                        "text": record.payload["text"],
                        "score": record.score
                    }
            )
            for record in result.points
        ]
