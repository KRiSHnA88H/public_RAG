from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from final import generate_answer

ROOT = Path(__file__).parent

app = FastAPI()


app.mount(
    "/static",
    StaticFiles(directory=ROOT / "templates"),
    name="static"
)


templates = Jinja2Templates(
    directory=ROOT / "templates"
)


class Question(BaseModel):

    question: str


@app.get("/")
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.post("/ask")
async def ask(data: Question):

    answer = generate_answer(
        data.question
    )

    return {
        "answer": answer
    }