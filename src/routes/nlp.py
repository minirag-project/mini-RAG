from fastapi import Depends, FastAPI , APIRouter ,UploadFile , status , Request
from fastapi.responses import JSONResponse
import os
from models.enums import ResponseEnums
from models.projectModel import ProjectModel
from models.chunkModel import ChunkModel
from controllers import NLPController 
from .schemas import NLPRequest , SearchRequest
from models.enums.ResponseEnums import ResponseStatus
from tqdm.auto import tqdm
nlp_router = APIRouter(
    prefix = "/api/v1/nlp",
    tags = ['api_v1','nlp']
)

@nlp_router.post("/index/push/{project_id}")
async def push_index(req: Request, project_id: int, push_request: NLPRequest):

    project_model = await ProjectModel.create_instance(
        db_conn=req.app.state.db_client
    )

    data_chunk_model = await ChunkModel.create_instance(
        db_conn=req.app.state.db_client
    )

    project = await project_model.get_project_or_create(project_id=project_id)

    if not project:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"signal": ResponseStatus.PROJECT_NOT_FOUND.value}
        )

    nlp_controller = NLPController(
        vector_db_client=req.app.state.vector_db_client,
        generation_client=req.app.state.generation_client,
        embedding_client=req.app.state.embedding_client,
        template_parser=req.app.state.template_parser
    )

    has_records = True
    page_no = 1
    idx = 0
    inserted_count = 0

    collection_name = nlp_controller.create_collection_name(project_id=project.project_id)

    _ = await nlp_controller.vector_db_client.create_collection(collection_name=collection_name, embedding_size=req.app.state.embedding_client.embedding_size , do_reset=push_request.do_reset)

    total_chunks_count = await data_chunk_model.get_total_chunks_count_by_project_id(project_id=project.project_id)
    pbar = tqdm(total=total_chunks_count, desc="Indexing Chunks",position = 0)

    while has_records:
        data_chunks = await data_chunk_model.get_data_chunks_by_project_id(project_id=project.project_id, page= page_no)
        print(f"Fetched {len(data_chunks)} chunks for project_id {project.project_id} on page {page_no}")
        
        if len(data_chunks):
            page_no += 1

        if len(data_chunks) == 0 or not data_chunks:
            has_records = False
            break

        chunks_ids = [data_chunk.chunk_id for data_chunk in data_chunks]
        print(f"Indexing {len(data_chunks)} chunks for project_id {project.project_id} with chunk_ids: {chunks_ids}")
        idx += len(data_chunks)

        is_inserted = await nlp_controller.index_into_vector_db(project=project, data_chunks=data_chunks, chunks_ids=chunks_ids)

        if not is_inserted:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"signal": ResponseStatus.FAILED_TO_INSERT_CHUNKS.value}
            )
        
        inserted_count += len(data_chunks)
        pbar.update(len(data_chunks))

    pbar.close()
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"signal": ResponseStatus.SUCCESSFULLY_INSERTED_CHUNKS.value, "inserted_count": inserted_count}
    )


@nlp_router.get("/index/pull/{project_id}")
async def pull_index(req: Request, project_id: int):
    project_model = await ProjectModel.create_instance(
        db_conn=req.app.state.db_client
    )

    project = await project_model.get_project_or_create(project_id=project_id)

    nlp_controller = NLPController(
        vector_db_client=req.app.state.vector_db_client,
        generation_client=req.app.state.generation_client,
        embedding_client=req.app.state.embedding_client,
        template_parser=req.app.state.template_parser
    )

    collection_info = await nlp_controller.get_vector_db_collection(project=project)

    if collection_info is None:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"signal": ResponseStatus.COLLECTION_NOT_FOUND.value}
        )
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"signal": ResponseStatus.COLLECTION_IS_FOUND.value, "collection_info": collection_info}
    )

@nlp_router.post("/index/search/{project_id}")
async def search_index(req: Request, project_id: int, search_request: SearchRequest):
    project_model = await ProjectModel.create_instance(
        db_conn=req.app.state.db_client
    )

    project = await project_model.get_project_or_create(project_id=project_id)

    nlp_controller = NLPController(
        vector_db_client=req.app.state.vector_db_client,
        generation_client=req.app.state.generation_client,
        embedding_client=req.app.state.embedding_client,
        template_parser=req.app.state.template_parser
    )

    search_results = await nlp_controller.search_vector_db(project=project, query_text=search_request.text, limit=search_request.limit)

    if search_results is None:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"signal": ResponseStatus.COLLECTION_NOT_FOUND.value}
        )
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"signal": ResponseStatus.COLLECTION_IS_FOUND.value, "search_results": [chunk.dict() for chunk in search_results]}
    )

@nlp_router.post("/index/answer/{project_id}")
async def generate_answer(req: Request, project_id: int, search_request: SearchRequest):
    project_model = await ProjectModel.create_instance(
        db_conn=req.app.state.db_client
    )

    project = await project_model.get_project_or_create(project_id=project_id)

    nlp_controller = NLPController(
        vector_db_client=req.app.state.vector_db_client,
        generation_client=req.app.state.generation_client,
        embedding_client=req.app.state.embedding_client,
        template_parser=req.app.state.template_parser
    )

    answer , full_prompt , chat_history =  await nlp_controller.generate_response(project=project, query=search_request.text, limit=search_request.limit)

    serialized_history = [
    {
        "role": content.role,
        "parts": [part.text for part in content.parts],
    }
    for content in chat_history
]

    if answer is None:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"signal": ResponseStatus.RAG_ANSWER_GENERATION_FAILED.value}
        )
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"signal": ResponseStatus.RAG_ANSWER_GENERATED.value, "answer": answer, "full_prompt": full_prompt, "chat_history": serialized_history}
    )
