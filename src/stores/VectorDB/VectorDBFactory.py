from .VectorDBProviders import QDrantDB , pgvectorDB
from .VectorDBEnums import VectorDBType, DistanceMetric
from controllers.BaseController import BaseController
from sqlalchemy.orm import sessionmaker

class VectorDBFactory:
    def __init__(self, config, db_client:str = None):
        self.config = config
        self.base_controller = BaseController()
        self.db_client = db_client

    def create_vector_db(self, provider_name: str):
        if provider_name == VectorDBType.QDrant.value:
            qdrant_db_client = self.base_controller.get_database_path(self.config.VECTOR_DB_PATH)
            return QDrantDB(db_client = qdrant_db_client,
                            distance_metric = self.config.VECTOR_DB_DISTANCE_METRIC,
                            default_vector_size = self.config.EMBEDDING_MODEL_SIZE,
                            index_threshold = self.config.VECTOR_DB_INDEX_THRESHOLD)
        elif provider_name == VectorDBType.PGVector.value:
            return pgvectorDB(db_client = self.db_client,
                            distance_metric = self.config.VECTOR_DB_DISTANCE_METRIC,
                            default_vector_size = self.config.EMBEDDING_MODEL_SIZE,
                            index_threshold = self.config.VECTOR_DB_INDEX_THRESHOLD)
        else:
            return None