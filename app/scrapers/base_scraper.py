from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


class BaseScraper:

    def get_driver(self):

        options = Options()

        options.add_argument("--headless=new")

        options.add_argument("--no-sandbox")

        options.add_argument("--disable-dev-shm-usage")

        options.add_argument("--window-size=1920,1080")

        return webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )