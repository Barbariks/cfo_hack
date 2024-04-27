from bs4 import BeautifulSoup as Soup
import requests
import pandas as pd

import csv

fake_headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

df = pd.read_excel('dataset.xlsx')

product_id = df['Название продукта ']
urls_col = df['Список вакансий']
cur_product = product_id[0]

with open('dataset.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    
    writer.writerow(['Название продукта', 'Описание'])

    for i in range(len(product_id)):
        if not pd.isna(product_id[i]):
            cur_product = product_id[i]

        if pd.isna(urls_col[i]):
            continue

        r = requests.get(urls_col[i], headers=fake_headers)
        soup = Soup(r.text, 'html.parser')
        test = soup.find('div', {'data-qa': 'vacancy-description'})

        if test is None:
            print(i)
            continue

        writer.writerow([cur_product, test.get_text(separator=' ')])