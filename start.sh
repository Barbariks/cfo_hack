#!/bin/sh

echo "Запуск приложения..."

# apt install python3-pip

pip install -r requirements.txt

(cd frontend && streamlit run main.py) & (cd backend && uvicorn main:app) && fg