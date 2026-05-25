import time
import pandas as pd

from bs4 import BeautifulSoup

from app.scrapers.base_scraper import BaseScraper
from app.utils.logger import logger


class KotakScraper(BaseScraper):

    URL = "https://www.bankbazaar.com/fixed-deposit/kotak-mahindra-bank-senior-citizen-fixed-deposit-rates.html"

    BANK_NAME = "Kotak Mahindra Bank"

    def scrape(self):

        logger.info("Starting Kotak BankBazaar scrape")

        driver = self.get_driver()

        try:

            # Open page
            driver.get(self.URL)

            # Wait for JS content to load
            time.sleep(8)

            html = driver.page_source

            driver.quit()

            soup = BeautifulSoup(
                html,
                "html.parser"
            )

            # Find table
            table = soup.find(
                "table",
                class_="w-full caption-bottom text-sm border"
            )

            if not table:

                logger.error("Table not found")

                return pd.DataFrame()

            tbody = table.find("tbody")

            rows = tbody.find_all("tr")

            data = []

            for row in rows:

                cols = row.find_all("td")

                if len(cols) >= 2:

                    tenure = cols[0].get_text(strip=True)

                    rate = cols[1].get_text(strip=True)

                    data.append([
                        tenure,
                        rate
                    ])

            # Create dataframe
            df = pd.DataFrame(
                data,
                columns=[
                    "Tenure",
                    "Senior Citizen Rate"
                ]
            )

            # Add bank name
            df["Bank"] = self.BANK_NAME

            logger.info(
                f"Kotak BankBazaar rows scraped: {len(df)}"
            )

            return df

        except Exception as e:

            logger.error(
                f"Kotak BankBazaar scraping failed: {str(e)}"
            )

            driver.quit()

            return pd.DataFrame()