from prometheus_client import Counter, Histogram, generate_latest , CONTENT_TYPE_LATEST
from fastapi import Response , Request , FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
import time

REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests', ['method', 'endpoint', 'http_status'])
REQUEST_LATENCY = Histogram('http_request_latency_seconds', 'HTTP Request Latency', ['method', 'endpoint'])

class prometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time

        REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, http_status=response.status_code).inc()
        REQUEST_LATENCY.labels(method=request.method, endpoint=request.url.path).observe(process_time)

        return response

def setup_prometheus(app: FastAPI):

    app.add_middleware(prometheusMiddleware)

    @app.get("/ronaldo_is_the_real_goat_1210", include_in_schema=False)
    async def metrics():
        
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)