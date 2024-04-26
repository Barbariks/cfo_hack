from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from py_pdf_parser.loaders import load_file as LoadPDF
from pathlib import Path
from bs4 import BeautifulSoup as Soup
import requests


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
    #check {pdf}
    with open(pdf_buffer_path, "wb") as file:
        file.write(pdf.file.read())
    #check {file}

    pdf_doc = LoadPDF(pdf_buffer_path)
    #check {pdf_doc}

    text = ' '.join([element.text() for element in pdf_doc.elements])
    with open('tmp.txt', 'w', encoding='utf-8') as file:
        file.write(text)

    return {"message": "success"}

@app.post("/process_url")
def process_url(request : Request, url : str):
    #check {url}

    r = requests.get(url, headers=fake_headers)
    #check {r}

    soup = Soup(r.text, 'html.parser')

    test = soup.get('vacancy-description')
    print(test)

    info = soup.find_all('strong')
    print(info)
    # text = info.text
    # with open('tmp.txt', 'w', encoding='utf-8') as file:
    #     file.write(text)

    return {'message': 'success'}
