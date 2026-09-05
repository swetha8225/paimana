"""PAIMANA FastAPI application. Serves the REST API and the built frontend."""
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import ARTIFACTS_DIR, DB_PATH, STATIC_DIR
from .api import router

app = FastAPI(title='PAIMANA API',
              description='Pro-active Analytics for Infrastructure Monitoring '
                          'and Assessment (National Analytics) - MoSPI '
                          'central-sector projects.',
              version='1.0.0')

app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'],
                   allow_headers=['*'])
app.include_router(router)


@app.get('/api/health')
def health():
    return {'status': 'ok', 'db': os.path.exists(DB_PATH),
            'models': os.path.exists(f'{ARTIFACTS_DIR}/models.joblib')}


if os.path.isdir(STATIC_DIR):
    app.mount('/assets', StaticFiles(directory=os.path.join(STATIC_DIR, 'assets')),
              name='assets')

    @app.get('/')
    def index():
        return FileResponse(os.path.join(STATIC_DIR, 'index.html'))

    @app.get('/{path:path}')
    def spa(path: str):
        f = os.path.join(STATIC_DIR, path)
        if os.path.isfile(f):
            return FileResponse(f)
        return FileResponse(os.path.join(STATIC_DIR, 'index.html'))
