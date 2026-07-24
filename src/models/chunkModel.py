from sqlalchemy.future import select

from .DataBaseModel import DataBaseModel
from .db_schemas import DataChunk
from .enums.DataBaseEnums import DataBaseEnums
from bson.objectid import ObjectId
from pymongo import InsertOne

from sqlalchemy import func, delete

class ChunkModel(DataBaseModel):
    def __init__(self,db_conn):
        super().__init__(db_conn)
        self.connection = db_conn

    @classmethod
    async def create_instance(cls, db_conn):
        instance = cls(db_conn)
        return instance

    async def create_chunk(self,chunk_data:DataChunk):
        async with self.connection() as session:
            async with session.begin():
                session.add(chunk_data)
            await session.commit()
            await session.refresh(chunk_data)

        return chunk_data
    
    async def get_chunk(self, chunk_id: str):
        async with self.connection() as session:
            async with session.begin():
                query = select(DataChunk).where(DataChunk.chunk_id == chunk_id)
                result = await session.execute(query)
                chunk_record = result.scalar_one_or_none()

        return chunk_record
    
    async def create_chunks_bulk(self, chunks: list , batch_size: int =100):
        async with self.connection() as session:
            async with session.begin():
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    session.add_all(batch)
            await session.commit()

        return len(chunks)
    
    async def delete_chunks_by_project_id(self, project_id: int):
        async with self.connection() as session:
            async with session.begin():
                delete_query = delete(DataChunk).where(DataChunk.chunk_project_id == project_id)
                result = await session.execute(delete_query)
            await session.commit()

        return result.rowcount
    
    async def get_data_chunks_by_project_id(self, project_id: int, page: int = 1, page_size: int = 50):
        async with self.connection() as session:
            async with session.begin():
                query = select(DataChunk).where(DataChunk.chunk_project_id == project_id).offset((page - 1) * page_size).limit(page_size)
                result = await session.execute(query)
                chunks = result.scalars().all()
        return chunks

    async def get_total_chunks_count_by_project_id(self, project_id: int):
        async with self.connection() as session:
            async with session.begin():
                query = select(func.count(DataChunk.chunk_id)).where(DataChunk.chunk_project_id == project_id)
                result = await session.execute(query)
                total_count = result.scalar_one()
        return total_count
