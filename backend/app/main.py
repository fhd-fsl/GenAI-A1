from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.services.model_manager import model_manager
from app.routers import health, task1, task2, task3, task4

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    model_manager.load_models()
    yield
    # Shutdown
    model_manager.sessions.clear()

app = FastAPI(title="RestoreAI API", lifespan=lifespan)

# Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(task1.router)
app.include_router(task2.router)
app.include_router(task3.router)
app.include_router(task4.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
