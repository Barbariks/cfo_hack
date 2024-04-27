from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from pdf import get_text_from_pdf, create_pdf_from_b64
from hh_parse import Parser
from text_classifier import predict_url

parser = Parser()
def lifespan(app: FastAPI):
    global parser
    yield
    del parser


app = FastAPI(lifespan=lifespan)
app.add_middleware( CORSMiddleware, allow_origins=['*'] )

@app.post("/process_text")
def process_text(request : Request):
    json = request.json()
    text = json['text_vac']

    output = predict_url(text)

    return {'message': 'success'}

@app.post("/process_pdf")
def process_pdf(request : Request):
    json = request.json()
    b64_file = json['file_content']

    pdf_path = create_pdf_from_b64(b64_file)

    text = get_text_from_pdf(pdf_path)

    output = predict_url(text)

    return {'message': 'success'}

@app.post("/process_url")
def process_url(request : Request, url : str):
    # json = request.json()
    # url = json['url_vac']

    try:
        text = parser.parse(url)

        output = predict_url(text)

        return {'message': output}
    except Exception as e:
        return {'error': e}