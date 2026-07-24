from fastapi import FastAPI
from routes import base_router, data_router , nlp_router
from helpers.config import get_settings
from contextlib import asynccontextmanager
from stores.LLM.LLMProviderFactory import LLMProviderFactory
from stores.VectorDB.VectorDBFactory import VectorDBFactory
from stores.LLM.templates.templates_parser import template_parser
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    postgres_url = f"postgresql+asyncpg://{settings.POSTGRES_USERNAME}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    app.state.db_engine = create_async_engine(postgres_url)
    app.state.db_client = sessionmaker(app.state.db_engine, class_=AsyncSession, expire_on_commit=False)


    llm_factory = LLMProviderFactory(config=settings)
    app.state.generation_client = llm_factory.create_provider(settings.GENERATION_BACKEND)
    app.state.generation_client.set_generator(settings.GENERATION_MODEL_ID)

    app.state.embedding_client = llm_factory.create_provider(settings.EMBEDDING_BACKEND)
    app.state.embedding_client.set_embedding(settings.EMBEDDING_MODEL_ID, settings.EMBEDDING_MODEL_SIZE)

    vector_db_factory = VectorDBFactory(config=settings , db_client=app.state.db_client)

    app.state.vector_db_client = vector_db_factory.create_vector_db(settings.VECTOR_DB_BACKEND)
    await app.state.vector_db_client.connect()

    app.state.template_parser = template_parser(language=settings.DEFAULT_LANGUAGE)


    yield
    await app.state.db_engine.dispose()
    await app.state.vector_db_client.disconnect()


app = FastAPI(lifespan=lifespan)

app.include_router(base_router)
app.include_router(data_router)
app.include_router(nlp_router)




