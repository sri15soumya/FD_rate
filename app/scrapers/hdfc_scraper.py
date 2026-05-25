import time
import pandas as pd

from bs4 import BeautifulSoup

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from app.scrapers.base_scraper import BaseScraper
from app.utils.logger import logger


class HDFCScraper(BaseScraper):

    URL = "https://www.hdfc.bank.in/fixed-deposit/fd-interest-rate"

    BANK_NAME = "HDFC Bank"

    def scrape(self):

        logger.info("Starting HDFC scrape")

        driver = self.get_driver()

        try:

            driver.get(self.URL)

            wait = WebDriverWait(driver, 20)

            # Wait until table rows appear
            wait.until(
                EC.presence_of_element_located(
                    (By.TAG_NAME, "table")
                )
            )

            time.sleep(5)

            html = driver.page_source

            soup = BeautifulSoup(
                html,
                "html.parser"
            )

            tables = soup.find_all("table")

            logger.info(
                f"Tables found: {len(tables)}"
            )

            data = []

            for table in tables:

                rows = table.find_all("tr")

                for row in rows:

                    cols = row.find_all("td")

                    if len(cols) >= 3:

                        tenure = cols[0].get_text(
                            strip=True
                        )

                        senior_rate = cols[2].get_text(
                            strip=True
                        )

                        if (
                            tenure
                            and "%" in senior_rate
                        ):

                            data.append([
                                tenure,
                                senior_rate
                            ])

            # Remove duplicates
            data = list(
                set(tuple(x) for x in data)
            )

            if not data:

                logger.error(
                    "No HDFC data extracted"
                )

                driver.save_screenshot(
                    "hdfc_debug.png"
                )

                with open(
                    "hdfc_page.html",
                    "w",
                    encoding="utf-8"
                ) as f:

                    f.write(html)

                logger.info(
                    "Saved debug files"
                )

                return pd.DataFrame()

            df = pd.DataFrame(
                data,
                columns=[
                    "Tenure",
                    "Senior Citizen Rate"
                ]
            )

            df["Bank"] = self.BANK_NAME

            logger.info(
                f"HDFC rows scraped: {len(df)}"
            )

            return df

        except Exception as e:

            logger.error(
                f"HDFC scraping failed: {str(e)}"
            )

            return pd.DataFrame()

        finally:

            driver.quit()