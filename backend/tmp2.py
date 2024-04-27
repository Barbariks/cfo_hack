import selenium
import selenium.webdriver
from selenium.webdriver.common.by import By

url = 'https://hh.ru/vacancy/95144367?query=%D0%A0%D0%B0%D0%B7%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D1%87%D0%B8%D0%BA+%D0%B8%D0%B3%D1%80+%D0%BD%D0%B0+Unreal+Engin&hhtmFrom=vacancy_search_list'

driver = selenium.webdriver.Chrome()
driver.get(url)

button = driver.find_element(By.XPATH, '/html/body/div[5]/div/div/div[4]/div[1]/div/div/div/div/div/div[3]/div/button')

button.click()

description = driver.find_element(By.XPATH, '/html/body/div[5]/div/div/div[4]/div[1]/div/div/div/div/div/div[4]/div/div/div[1]/div')

print(description.text)