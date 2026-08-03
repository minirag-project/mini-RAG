from fastapi import Depends, FastAPI , APIRouter
import os
from helpers.config import get_settings, Settings
from datetime import datetime


base_router = APIRouter( prefix = "/api/v1" , tags =['api_v1'])

@base_router.get("/")
async def default(settings: Settings = Depends(get_settings)):
    app_name = settings.APP_NAME
    app_version = settings.APP_VERSION
    app_description = settings.APP_DESCRIPTION
    current_date = datetime.now().date()
    return {'message': f"Welcome to the {app_name} API!", 'version': app_version, 'description': app_description, 'date': current_date}