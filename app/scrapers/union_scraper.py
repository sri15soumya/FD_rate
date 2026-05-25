import time
import pandas as pd

from bs4 import BeautifulSoup

from app.scrapers.base_scraper import BaseScraper
from app.utils.logger import logger


class UnionBankScraper(BaseScraper):

    URL = "https://www.etmoney.com/fixed-deposit/union-bank-fd-rates/14"

    BANK_NAME = "Union Bank"

    def scrape(self):

        logger.info("Starting Union Bank scrape")

        driver = self.get_driver()

        try:

            driver.get(self.URL)

            time.sleep(8)

            html = driver.page_source

            driver.quit()

            soup = BeautifulSoup(
                html,
                'html.parser'
            )

            table = soup.find(
                'table',
                id='t01'
            )

            data = []

            if table:

                rows = table.find_all('tr')[1:]

                for row in rows:

                    cols = row.find_all('td')

                    if len(cols) == 3:

                        tenure = cols[0].get_text(strip=True)

                        senior_rate = cols[2].get_text(strip=True)

                        data.append([
                            tenure,
                            senior_rate
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
                f"Union Bank rows scraped: {len(df)}"
            )

            return df

        except Exception as e:

            logger.error(
                f"Union Bank scraping failed: {str(e)}"
            )

            driver.quit()

            return pd.DataFrame()