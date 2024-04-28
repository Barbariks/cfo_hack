from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pdf import get_text_from_pdf, create_pdf_from_b64
from hh_parse import Parser
from text_classifier import ensemble_predict
from course_config import get_course_data

class URLItem(BaseModel):
    url_vac : str

parser = Parser()
def lifespan(app: FastAPI):
    global parser
    yield
    del parser


app = FastAPI(lifespan=lifespan)
app.add_middleware( CORSMiddleware, allow_origins=['*'] )

@app.post("/process_text")
async def process_text(request : Request):
    json = await request.json()
    text = json['text_vac']

    output = ensemble_predict(text)

    return get_course_data(output)

@app.post("/process_pdf")
async def process_pdf(request : Request):
    json = await request.json()
    b64_file = json['file_content']

    pdf_path = create_pdf_from_b64(b64_file)

    text = get_text_from_pdf(pdf_path)

    output = ensemble_predict(text)

    return get_course_data(output)

@app.post("/process_url")
async def process_url(request : Request):
    json = await request.json()
    url = json['url_vac']

    text = parser.parse(url)
    output = ensemble_predict(text)

    return get_course_data(output)