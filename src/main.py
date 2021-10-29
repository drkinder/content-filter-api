from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from web_widget.routes import router as web_widget_router

app = FastAPI(
    title="MADS Capstone - Social Media Content Filter",
    version="1.0.0",
)


app.include_router(web_widget_router)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "Access-Control-Allow-Origin"],
)
