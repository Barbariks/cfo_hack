#!/bin/bash

echo "Запуск приложения..."

apt install python3-pip

cd /backend
pip install -r requirements.txt
uvicorn main:app --port 8000
cd ..

streamlit run frontend/main.py