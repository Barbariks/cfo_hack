from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from py_pdf_parser.loaders import load_file as LoadPDF
from pathlib import Path
import validators
import selenium.webdriver
from selenium.webdriver.common.by import By

buffer_directory = "pdf_buffer/"
pdf_buffer_path = buffer_directory + "file.pdf"

xpath_vacancy_archive_button = '//*[@id="HH-React-Root"]/div/div/div[4]/div[1]/div/div/div/div/div/div[3]/div/button'
xpath_vacancy_description = '//*[@id="HH-React-Root"]/div/div/div[4]/div[1]/div/div/div/div/div/div[4]/div/div/div[1]/div'

Path(buffer_directory).mkdir(exist_ok=True) 

app = FastAPI()
app.add_middleware( CORSMiddleware, allow_origins=['*'] )

driver = selenium.webdriver.Firefox()

@app.post("/process_text")
def process_text(request : Request, text : str):
    return {'message': text}

@app.post("/process_pdf")
def process_pdf(request : Request, pdf : UploadFile = File(...)):
    with open(pdf_buffer_path, "wb") as file:
        file.write(pdf.file.read())

    pdf_doc = LoadPDF(pdf_buffer_path)

    text = ' '.join([element.text() for element in pdf_doc.elements])

    return {'message': text}

@app.post("/process_url")
def process_url(request : Request, url : str):
    if not validators.url(url):
        return {'error': 'invalid url'}

    try:
        driver.get(url)
    except:
        return {'error': 'error while processing url'}
    
    try:
        button = driver.find_element(By.XPATH, xpath_vacancy_archive_button)
        button.click()
    except:
        pass

    try:
        description = driver.find_element(By.XPATH, xpath_vacancy_description)

        text = description.text
    except:
        return {'error': 'could not find vacancy description'}

    return {'message': text}
