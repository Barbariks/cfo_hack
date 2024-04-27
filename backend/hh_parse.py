import selenium.webdriver
from selenium.webdriver.common.by import By
import validators

xpath_vacancy_archive_button    = '//*[@id="HH-React-Root"]/div/div/div[4]/div[1]/div/div/div/div/div/div[3]/div/button'
xpath_vacancy_description       = '//*[@id="HH-React-Root"]/div/div/div[4]/div[1]/div/div/div/div/div/div[4]/div/div/div[1]/div'

class Parser():
    def __init__(self):
        self.driver = selenium.webdriver.Firefox()

    def __del__(self):
        self.driver.quit()

    def parse(self, url : str):
        if not validators.url(url):
            raise Exception(f'invalid url: {url}')
        
        try:
            self.driver.get(url)
        except:
            raise Exception(f'error while processing url: {url}')

        try:
            button = self.driver.find_element(By.XPATH, xpath_vacancy_archive_button)
            button.click()
        except:
            pass

        try:
            description = self.driver.find_element(By.XPATH, xpath_vacancy_description)

            return description.text
        except:
            raise Exception(f'could not find vacancy description at {url}')
