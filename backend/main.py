from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from py_pdf_parser.loaders import load_file as LoadPDF
from pathlib import Path
from bs4 import BeautifulSoup as Soup
import requests
import validators


buffer_directory = "pdf_buffer/"
pdf_buffer_path = buffer_directory + "file.pdf"

fake_headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

Path(buffer_directory).mkdir(exist_ok=True) 

app = FastAPI()
app.add_middleware( CORSMiddleware, allow_origins=['*'] )

@app.post("/process_text")
def process_text(request : Request, text : str):
    return {'message': 'success'}

@app.post("/process_pdf")
def process_pdf(request : Request, pdf : UploadFile = File(...)):
    with open(pdf_buffer_path, "wb") as file:
        file.write(pdf.file.read())

    pdf_doc = LoadPDF(pdf_buffer_path)

    text = ' '.join([element.text() for element in pdf_doc.elements])

    return {"message": "success"}

@app.post("/process_url")
def process_url(request : Request, url : str):
    if not validators.url(url):
        return {'error': 'invalid url'}

    try:
        r = requests.get(url, headers=fake_headers)
    except:
        return {'error': f'could not make request for {url}'}
    
    if r.status_code != 200:
        return {'error': f'status code {r.status_code}'}

    soup = Soup(r.text, 'html.parser')

    test = soup.find('div', {'data-qa': 'vacancy-description'})

    if test is None:
        return {'error': 'invalid url page'}
    
    text = test.get_text(separator=' ')

    return {'message': 'success'}
