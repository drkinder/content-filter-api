from fastapi import Request

from .filters import random_filter
from .models import FilteredContent


async def filter_twitter_content(request: Request) -> FilteredContent:
    data = await request.json()
    twitter_body = data.get('body')
    return FilteredContent(filter=random_filter(data.get('threshold', 0)))
