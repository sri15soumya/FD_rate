import time
import pandas as pd

from bs4 import BeautifulSoup

from app.scrapers.base_scraper import BaseScraper
from app.utils.logger import logger


class SBIScraper(BaseScraper):

    URL = "https://sbi.bank.in/web/interest-rates/deposit-rates/retail-domestic-term-deposits"

    BANK_NAME = "SBI"

    def scrape(self):

        logger.info("Starting SBI scrape")

        driver = self.get_driver()

        try:

            driver.get(self.URL)

            time.sleep(8)

            html = driver.page_source

            driver.quit()

            soup = BeautifulSoup(
                html,
                "html.parser"
            )

            table = soup.find(
                "table",
                {"class": "table table-bordered"}
            )

            rows = table.find_all("tr")[2:]

            data = []

            for row in rows:

                cols = row.find_all("td")

                if cols:

                    tenor = cols[0].get_text(strip=True)

                    rate = cols[4].get_text(strip=True)

                    data.append([
                        tenor,
                        rate
                    ])

            df = pd.DataFrame(
                data,
                columns=[
                    "Tenure",
                    "Senior Citizen Rate"
                ]
            )

            df['Bank'] = self.BANK_NAME

            logger.info(
                f"SBI rows scraped: {len(df)}"
            )

            return df

        except Exception as e:

            logger.error(
                f"SBI scraping failed: {str(e)}"
            )

            driver.quit()

            return pd.DataFrame()