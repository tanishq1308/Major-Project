from pydantic import BaseModel


class AppDetails(BaseModel):
    appname: str
    version: str
    email: list
    author: list


class PromptQuery(BaseModel):
    prompt: str


class OutfitResponse(BaseModel):
    request: list