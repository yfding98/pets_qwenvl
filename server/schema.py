from pydantic import BaseModel


class RequestData(BaseModel):
    prompt: str
    images: list = []
