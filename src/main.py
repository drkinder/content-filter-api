from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from content_filter.routes import (filter_twitter_content)

app = FastAPI(
    title="MADS Capstone - Social Media Content Filter",
    version="1.0.0",
)


app.add_api_route('/filter-twitter-content/', filter_twitter_content, methods=['POST'])


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "Access-Control-Allow-Origin"],
)
