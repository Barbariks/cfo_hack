#!/bin/bash

echo "Запуск приложения..."

apt install python3-pip

pip install -r requirements.txt
cd /backend
uvicorn main:app --port 8000
cd ..

streamlit run frontend/main.py