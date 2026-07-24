import json

from ..VectorDBInterface import VectorDBInterface
from ..VectorDBEnums import pgvectorTableSchema , pgvectorDistanceMethodEnums , pgvectorIndexTypeEnums
import logging
from typing import List
from models.db_schemas import retrievedchunk
from sqlalchemy.sql import text as sql_text

class pgvectorDB(VectorDBInterface):
    def __init__(self, db_client : str , distance_metric : str , default_vector_size : int = 786, index_type : str = pgvectorIndexTypeEnums.HNSW.value , index_threshold : int = 100):
        self.client = db_client
        self.distance_metric = distance_metric
        self.default_vector_size = default_vector_size
        self.index_type = index_type
        self.index_threshold = index_threshold
        self.default_index_name = lambda collection_name : f"{collection_name}_vector_idx"

        if self.distance_metric == "Cosine":
            self.distance_metric = pgvectorDistanceMethodEnums.COSINE.value
        elif self.distance_metric == "Dot":
            self.distance_metric = pgvectorDistanceMethodEnums.DOT.value

        self.table_prefix = pgvectorTableSchema._PREFIX.value
        self.logger = logging.getLogger('uvicorn')

    async def connect(self):
        async with self.client() as session:
            async with session.begin():
                await session.execute(sql_text("CREATE EXTENSION IF NOT EXISTS vector"))

            await session.commit()

    async def disconnect(self):
        pass

    async def is_collection_exists(self, collection_name:str) -> bool:
        async with self.client() as session:
            async with session.begin():
                list_tbl = sql_text('SELECT * FROM pg_tables WHERE tablename = :collection_name')
                session_result = await session.execute(list_tbl, {'collection_name': f"{collection_name}"})
                record = session_result.scalar_one_or_none()
            
        return record 

    async def list_all_collections(self) -> List:
        async with self.client() as session:
            async with session.begin():
                list_tbl = sql_text('SELECT tablename FROM pg_tables WHERE tablename LIKE :prefix')
                session_result = await session.execute(list_tbl, {'prefix': f"{self.table_prefix}"})
                records = session_result.scalars().all()
        return records
    
    async def get_collection_info(self, collection_name:str):
        async with self.client() as session:
            async with session.begin():
                table_info_sql = sql_text(f'''SELECT schemaname, tablename, tableowner, tablespace, hasindexes 
                                        FROM pg_tables WHERE tablename = '{collection_name}'
                ''')
                count_sql = sql_text(f'SELECT COUNT(*) FROM {collection_name}')

                table_info = await session.execute(table_info_sql)
                table_count = await session.execute(count_sql)

                table_data = table_info.fetchone()
                if not table_data:
                    return None
                
                return {
                    "table_info" : {
                        
                        "schemaname" : table_data.schemaname,
                        "tablename" : table_data.tablename,
                        "tableowner" : table_data.tableowner,
                        "tablespace" : table_data.tablespace,
                        "hasindexes" : table_data.hasindexes
                    },
                    "table_count" : table_count.scalar_one()
                }
            
    
    async def delete_collection(self, collection_name:str):
        async with self.client() as session:
            async with session.begin():
                self.logger.info(f"Dropping collection {collection_name}...")
                drop_table_sql = sql_text(f'DROP TABLE IF EXISTS {collection_name}')
                await session.execute(drop_table_sql)
                await session.commit()

            
            return True
        
    async def is_index_exists(self, collection_name:str) -> bool:

        index_name = self.default_index_name(collection_name)
        async with self.client() as session:
            async with session.begin():
                check_index_sql = sql_text('SELECT 1 FROM pg_indexes WHERE tablename = :collection_name AND indexname = :index_name')
                result = await session.execute(check_index_sql, {'collection_name': f"{collection_name}", 'index_name': f"{index_name}"})
                return bool(result.scalar_one_or_none())
            
    async def create_index(self, collection_name:str, index_type:str = pgvectorIndexTypeEnums.HNSW.value):

        index_name = self.default_index_name(collection_name)
        
        if await self.is_index_exists(collection_name):
            self.logger.warning(f"Index {index_name} already exists for collection {collection_name}. Skipping creation.")
            return False
        
        async with self.client() as session:
            async with session.begin():
                count_sql = sql_text(f'SELECT COUNT(*) FROM {collection_name}')
                result = await session.execute(count_sql)
                record_count = result.scalar_one()

                if record_count < self.index_threshold:
                    self.logger.warning(f"Collection {collection_name} has {record_count} records, which is below the index threshold of {self.index_threshold}. Skipping index creation.")
                    return False
                
                self.logger.info(f"Creating index {index_name} for collection {collection_name}...")
                create_index_sql = sql_text(f'CREATE INDEX {index_name} ON {collection_name} '
                                            f'USING {index_type} ({pgvectorTableSchema.VECTOR.value} {self.distance_metric})')
                await session.execute(create_index_sql)

                self.logger.info(f"END: Created vector index for collection: {collection_name}")
            
    async def reset_index(self, collection_name:str):
        if not await self.is_index_exists(collection_name):
            self.logger.warning(f"Index does not exist for collection {collection_name}. Skipping reset.")
            return False
        
        async with self.client() as session:
            async with session.begin():
                index_name = self.default_index_name(collection_name)
                self.logger.info(f"Dropping index {index_name} for collection {collection_name}...")
                drop_index_sql = sql_text('DROP INDEX IF EXISTS :index_name')
                await session.execute(drop_index_sql, {'index_name': f"{index_name}"})
                await session.commit()

                return await self.create_index(collection_name, self.index_type)


    async def create_collection(self, collection_name:str, embedding_size:int, do_reset:bool = False):
        if do_reset:
            await self.delete_collection(collection_name)

        if await self.is_collection_exists(collection_name):
            self.logger.warning(f"Collection {collection_name} already exists. Skipping creation.")
            return False
        
        async with self.client() as session:
            async with session.begin():
                self.logger.info(f"Creating collection {collection_name} with embedding size {embedding_size}...")
                create_table_sql = sql_text(f'CREATE TABLE {collection_name} ('
                                            f'{pgvectorTableSchema.ID.value} bigserial PRIMARY KEY, '
                                            f'{pgvectorTableSchema.TEXT.value} text, '
                                            f'{pgvectorTableSchema.VECTOR.value} vector({embedding_size}), '
                                            f'{pgvectorTableSchema.METADATA.value} jsonb DEFAULT \'{{}}\', '
                                            f'{pgvectorTableSchema.CHUNK_ID.value} integer,'
                                            f'FOREIGN KEY ({pgvectorTableSchema.CHUNK_ID.value}) REFERENCES data_chunks(chunk_id) ON DELETE CASCADE'
                                            ')')
                await session.execute(create_table_sql)
                await session.commit()

                return True
            
    async def insert_collection(self, collection_name:str, text:str, vector:list, metadata:dict = None, record_id:str = None):
        
        is_collection_exists = await self.is_collection_exists(collection_name)
        if not is_collection_exists:
            self.logger.warning(f"Collection {collection_name} does not exist. Skipping insertion.")
            return False

        if not record_id:
            self.logger.warning(f"Record ID is required for insertion. Skipping insertion.")
            return False
        
        async with self.client() as session:
            async with session.begin():
                metadata_json = json.dumps(metadata) if metadata else '{}'
                self.logger.info(f"Inserting record into collection {collection_name}...")
                insert_sql = sql_text(f'INSERT INTO :collection_name ({pgvectorTableSchema.TEXT.value}, {pgvectorTableSchema.VECTOR.value}, {pgvectorTableSchema.METADATA.value}, {pgvectorTableSchema.CHUNK_ID.value}) VALUES (:text, :vector, :metadata, :chunk_id)')
                await session.execute(insert_sql, {'collection_name': f"{collection_name}",
                                                    'text': text,
                                                    'vector': "[" + ", ".join(str(x) for x in vector) + "]",
                                                    'metadata': metadata_json,
                                                    'chunk_id': record_id})
                await session.commit()

        await self.create_index(collection_name, self.index_type)

        return True
        


    async def insert_collection_batch(self, collection_name, texts, vectors, metadatas=None, record_ids=None, batch_size=20):

        is_collection_exists = await self.is_collection_exists(collection_name)
        if not is_collection_exists:
            self.logger.warning(f"Collection {collection_name} does not exist. Skipping batch insertion.")
            return False

        if len(vectors) != len(record_ids):
            self.logger.warning("Vectors and record_ids must have the same length. Skipping batch insertion.")
            return False

        if metadatas is None or len(metadatas) == 0:
            metadatas = [{}] * len(vectors)

        async with self.client() as session:
            self.logger.info(f"Inserting batch of records into collection {collection_name}...")
            for i in range(0, len(vectors), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_vectors = vectors[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                batch_record_ids = record_ids[i:i + batch_size]

                values = []

                for _text, _vector, _metadata, _chunk_id in zip(batch_texts, batch_vectors, batch_metadatas, batch_record_ids):
                    metadata_json = json.dumps(_metadata, ensure_ascii=False) if _metadata is not None else "{}"
                    values.append({
                        'text': _text,
                        'vector': "[" + ", ".join(str(x) for x in _vector) + "]",
                        'metadata': metadata_json,
                        'chunk_id': _chunk_id
                    })

                batch_insert_sql = sql_text(
                    f'INSERT INTO {collection_name} '
                    f'({pgvectorTableSchema.TEXT.value}, {pgvectorTableSchema.VECTOR.value}, '
                    f'{pgvectorTableSchema.METADATA.value}, {pgvectorTableSchema.CHUNK_ID.value}) '
                    f'VALUES (:text, :vector, :metadata, :chunk_id)'
                )

                await session.execute(batch_insert_sql, values)
                await session.commit()

        await self.create_index(collection_name, self.index_type)

        return True
            
    async def search_collection_by_vector(self, collection_name:str , vector:list , limit:int = 10) -> List[retrievedchunk]:

        if not await self.is_collection_exists(collection_name):
            self.logger.warning(f"Collection {collection_name} does not exist. Skipping batch insertion.")
            return False
        
        vector = "[" + ", ".join(str(x) for x in vector) + "]"
        async with self.client() as session:
            async with session.begin():
                self.logger.info(f"Searching collection {collection_name} by vector...")
                search_sql = sql_text(f"SELECT {pgvectorTableSchema.TEXT.value} as text ,1 - ({pgvectorTableSchema.VECTOR.value} <=> :vector) as score "
                                    f"FROM {collection_name} "
                                    f"ORDER BY score DESC "
                                    f"LIMIT :limit")
                session_result = await session.execute(search_sql, {'vector': vector, 'limit': limit})
                records = session_result.fetchall()
                
                return [retrievedchunk(**{
                    "text" : record.text,
                    "score" : record.score
                }) for record in records]
