from fastapi_utils.api_model import APIModel


class FilteredContent(APIModel):
    html_content: str
