from bson import ObjectId
from fastapi import Depends, FastAPI , APIRouter ,UploadFile , status , Request
from fastapi.responses import JSONResponse
import os
from models.projectModel import ProjectModel
from models.chunkModel import ChunkModel
from helpers.config import get_settings, Settings
from controllers import DataController , ProjectController , ProcessController , NLPController
import aiofiles
import models
import logging
from .schemas import processData 
from models.db_schemas import DataChunk , File
from models.FilesModel import FilesModel
from models.enums.FileEnums import FileEnums


logger = logging.getLogger('uvicorn.error')

data_router = APIRouter(
    prefix = "/api/v1/data",
    tags = ['api_v1','data']
)

def clean_string(text: str) -> str:
    return (
        text.replace("\x00", "")
            .encode("utf-8", "ignore")
            .decode("utf-8", "ignore")
    )

def clean_metadata(obj):
    if isinstance(obj, dict):
        return {k: clean_metadata(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_metadata(v) for v in obj]
    elif isinstance(obj, str):
        return clean_string(obj)
    return obj

@data_router.post("/process/{project_id}")
async def process_data(req: Request, project_id: int, file: UploadFile,
                    settings: Settings = Depends(get_settings)):
    app_settings = get_settings()
    
    project_model = await ProjectModel.create_instance(
        db_conn=req.app.state.db_client
    )

    project = await project_model.get_project_or_create(project_id=project_id)

    is_valid , message = DataController().is_valid_file(file = file)
    if not is_valid:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"message": message})
    
    folder_path = ProjectController().get_file_path(project_id = project_id)
    filename = DataController().generate_unique_filename(file.filename, project_id)
    file_location = os.path.join(folder_path, filename)

    while os.path.exists(file_location):
        filename = DataController().generate_unique_filename(file.filename, project_id)
        file_location = os.path.join(folder_path, filename)

    try:
        async with aiofiles.open(file_location, 'wb') as out_file:
            while chunk := await file.read(app_settings.FILE_CHUNK_SIZE):
                await out_file.write(chunk)
    except Exception as e:
        logger.error(f"Error occurred while saving the file: {str(e)}")
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=models.ResponseStatus.File_Upload_Failed.value)
    
    file_model = await FilesModel.create_instance(
        db_conn=req.app.state.db_client
    )

    file = File(
        file_name = filename,
        file_project_id = project.project_id,
        file_size = os.path.getsize(file_location),
        file_type = FileEnums.FILE.value
    )
    file_record = await file_model.create_file(file_data=file)

    return JSONResponse(status_code=status.HTTP_200_OK, content={'signal' : models.ResponseStatus.File_Upload_Success.value,
                        'file_id': str(file_record.file_id) , 'file_project_id': str(project.project_id)})
    

@data_router.post("/process_file/{project_id}")
async def process_file(req: Request, project_id: int, data: processData):
    chunk_size = data.chunk_size
    overlap = data.overlap
    reset = data.reset

    project_model = await ProjectModel.create_instance(
        db_conn=req.app.state.db_client
    )

    project = await project_model.get_project_or_create(project_id=project_id)

    projects_files_ids = {}

    file_model = await FilesModel.create_instance(
        db_conn=req.app.state.db_client
    )

    if data.file_id:
        file_record = await file_model.get_file_by_filename(file_project_id=project.project_id, file_name=data.file_id)
        if file_record is None:
            return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"signal": models.ResponseStatus.File_Not_Found.value})
        
        projects_files_ids = {
            file_record.file_id : file_record.file_name
        }
        
    else:
        projects_files = await file_model.get_all_project_files(file_project_id=project.project_id, file_type=FileEnums.FILE.value)
        projects_files_ids = {
            record.file_id : record.file_name
            for record in projects_files
        }

    if  len(projects_files_ids) == 0:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"signal": models.ResponseStatus.No_Chunks_Found.value})



    process_controller = ProcessController(project_id=project_id)

    no_records = 0
    no_files = 0

    nlp_controller = NLPController(
        vector_db_client=req.app.state.vector_db_client,
        generation_client=req.app.state.generation_client,
        embedding_client=req.app.state.embedding_client,
        template_parser=req.app.state.template_parser
    )

    chunk_model = await ChunkModel.create_instance(
        db_conn=req.app.state.db_client
    )

    if reset == 1:
        collection_name = nlp_controller.create_collection_name(project_id=project.project_id)
        _ = await nlp_controller.vector_db_client.delete_collection(collection_name = collection_name)

        _ = await chunk_model.delete_chunks_by_project_id(
                project_id=project.project_id
            )
        

    for file_id, file_name in projects_files_ids.items():

        file_content = process_controller.get_file_content(project_id=project_id, filename=file_name)

        if file_content is None:
            logger.error(f"File content is None for file_id: {file_id} in project_id: {project_id}")
            continue
        
        chunks = process_controller.process_file_content(file_content=file_content, file_id=file_name,
                                                            chunk_size=chunk_size, chunk_overlap=overlap)
        
        if chunks is None or len(chunks) == 0:
            return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"signal": models.ResponseStatus.File_processing_failed.value})
        
        chunks_record = [
            DataChunk(
                chunk_text=clean_string(chunk.page_content),
                chunk_metadata=clean_metadata(chunk.metadata),
                chunk_order=i + 1,
                chunk_project_id=project.project_id,
                chunk_file_id=file_id,
            )
            for i, chunk in enumerate(chunks)
        ]
        no_records += await chunk_model.create_chunks_bulk(chunks=chunks_record)
        no_files += 1

    return JSONResponse(status_code=status.HTTP_200_OK, content={"signal": models.ResponseStatus.File_Processed_Success.value, "inserted_records": no_records, "processed_files": no_files})