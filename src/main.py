from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from content_filter.routes import router as content_filter_router

app = FastAPI(
    title="MADS Capstone - Social Media Content Filter",
    version="1.0.0",
)


app.include_router(content_filter_router)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "Access-Control-Allow-Origin"],
)
