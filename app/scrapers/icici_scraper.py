import pandas as pd

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

from app.scrapers.base_scraper import BaseScraper
from app.utils.logger import logger


class ICICIScraper(BaseScraper):

    URL = "https://www.paisabazaar.com/fixed-deposit/icici-bank-fd-rates/"

    BANK_NAME = "ICICI Bank"

    def scrape(self):

        logger.info("Starting ICICI PaisaBazaar scrape")

        driver = self.get_driver()

        try:

            driver.get(self.URL)

            wait = WebDriverWait(driver, 30)

            #  Wait until the FD table row with "7 days to 45 days" is present
            wait.until(
                EC.presence_of_element_located(
                    (By.XPATH, "//td[contains(text(), '7 days to 45 days')]")
                )
            )

            logger.info("FD rate table detected on page")

            # Hand off to BeautifulSoup for reliable parsing
            soup = BeautifulSoup(driver.page_source, "html.parser")

            driver.quit()

            #  Find the table containing "7 days to 45 days"
            target_table = None

            for table in soup.find_all("table"):
                if table.find(
                    "td",
                    string=lambda t: t and "7 days to 45 days" in t
                ):
                    target_table = table
                    break

            if not target_table:
                logger.error("Could not find FD rates table in page source")
                return pd.DataFrame()

            rows = target_table.find("tbody").find_all("tr")

            logger.info(f"Total rows found in FD table: {len(rows)}")

            data = []

            for row in rows:

                # Skip header rows containing <th> or <strong> only rows
                if row.find("th") or row.find("strong"):
                    continue

                cols = row.find_all("td")

                #  Need 3 cols: Tenure | Regular | Senior Citizens
                if len(cols) < 3:
                    continue

                tenure = cols[0].get_text(strip=True)

                if not tenure:
                    continue

                # cols[2] = Senior Citizens rate (plain text, no spans)
                senior_rate = cols[2].get_text(strip=True)

                logger.info(
                    f"  Row scraped → Tenure: {tenure} | Senior Rate: {senior_rate}"
                )

                data.append([tenure, senior_rate])

            df = pd.DataFrame(
                data,
                columns=[
                    "Tenure",
                    "Senior Citizen Rate"
                ]
            )

            df["Bank"] = self.BANK_NAME

            logger.info(f"ICICI PaisaBazaar rows scraped: {len(df)}")

            return df

        except Exception as e:

            logger.error(f"ICICI PaisaBazaar scraping failed: {str(e)}")

            driver.quit()

            return pd.DataFrame()