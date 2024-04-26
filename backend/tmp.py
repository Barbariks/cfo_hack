from bs4 import BeautifulSoup as Soup
import requests

fake_headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
url = 'https://hh.ru/vacancy/96176670?query=%D0%BF%D1%80%D0%BE%D0%B3%D1%80%D0%B0%D0%BC%D0%BC%D0%B8%D1%81%D1%82+1C&hhtmFrom=vacancy_search_list'

r = requests.get(url, headers=fake_headers)

soup = Soup(r.text, 'html.parser')

test = soup.find('div', {'data-qa': 'vacancy-description'})
print(test.find_all('strong'))