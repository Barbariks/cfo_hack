***Команда Ботики и наше решение кейса на хакатоне ЦФО***

Чтобы корректно запустить наше решение надо скачать все зависимости, так же эта строчка прописана в bash скрипте, по этому она не обязательна на данном этапе

```pip install -r requirements.txt```

**Windows**

```cd /backend && uvicorn main:app --port 8000 && cd .. && streamlit run frontend/main.py```

**Unix/Linux**

Далее запускаем bash файл для Unix систем
```./strart.sh```
