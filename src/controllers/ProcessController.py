from .BaseController import BaseController
from .ProjectController import ProjectController
import os
from langchain_community.document_loaders import PyMuPDFLoader , TextLoader
from models import ProcessEnums
from dataclasses import dataclass
from typing import List

@dataclass
class document:
    page_content: str
    metadata: dict


class ProcessController(BaseController):
    def __init__(self,project_id:str):
        self.projectpath = ProjectController().get_file_path(project_id=project_id)

    def get_file_extension(self,filename:str):
        return os.path.splitext(filename)[-1]
    
    def get_file_loader(self, filename:str):
        file_extension = self.get_file_extension(filename)
        file_path = os.path.join(
            self.projectpath,
            filename
        )

        if os.path.exists(file_path) is False:
            return None
        
        if file_extension == ProcessEnums.PDF.value:
            return PyMuPDFLoader(file_path)
        elif file_extension == ProcessEnums.TXT.value:
            return TextLoader(
                        file_path,
                        encoding="utf-8"
                    )
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
    def get_file_content(self , project_id:str , filename:str):
        loader = self.get_file_loader(filename)
        if loader:
            return loader.load()
        else:
            return None

    def process_file_content(self, file_content: list, file_id:str,
                            chunk_size: int = 100, chunk_overlap: int = 20):
        
        file_content_texts = [
            rec.page_content
            for rec in file_content
        ]
        file_metadata = [
            rec.metadata
            for rec in file_content
        ]

        chunks = self.process_simple_splitter(
            file_content_texts,
            file_metadata,
            chunk_size=chunk_size
        )

        return chunks

    def process_simple_splitter(self, chunks: List[str], metadatas: List[dict],
                                chunk_size: int = 100, splitter_tag: str = "\n"):

        line_meta_pairs = []
        for text, meta in zip(chunks, metadatas):
            for line in text.split(splitter_tag):
                line = line.strip()
                if line:
                    line_meta_pairs.append((line, meta))

        result_chunks = []
        current_chunk = ""
        current_metadatas = []

        for line, meta in line_meta_pairs:
            current_chunk += line + splitter_tag
            current_metadatas.append(meta)

            if len(current_chunk) >= chunk_size:
                result_chunks.append(
                    document(
                        page_content=current_chunk.strip(),
                        metadata=self._merge_metadata(current_metadatas)
                    )
                )
                current_chunk = ""
                current_metadatas = []

        if current_chunk:
            result_chunks.append(
                document(
                    page_content=current_chunk.strip(),
                    metadata=self._merge_metadata(current_metadatas)
                )
            )

        return result_chunks

    def _merge_metadata(self, metadatas: List[dict]) -> dict:
        if not metadatas:
            return {}
        merged = dict(metadatas[0])
        pages = [m.get("page") for m in metadatas if m.get("page") is not None]
        if pages:
            merged["page_start"] = min(pages)
            merged["page_end"] = max(pages)
        return merged