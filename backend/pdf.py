from pathlib import Path
from py_pdf_parser.loaders import load_file as LoadPDF
from base64 import b64decode


pdf_directory_path = "pdf_buffer/"
Path(pdf_directory_path).mkdir(exist_ok=True)

def get_text_from_pdf(file_path: str):
    pdf = LoadPDF(file_path)
    return ' '.join([element.text() for element in pdf.elements])

def create_pdf_from_b64(bytes, file_name: str = 'file.pdf'):
    file_path = pdf_directory_path + file_name
    with open(file_path, "wb") as file:
        file.write(b64decode(bytes))
        return file_path
        