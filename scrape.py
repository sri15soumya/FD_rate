


# =======================
# IMPORTS AND SETUP
# =======================
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pandas as pd
import numpy as np
import re
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# =======================
# FLASK + DATABASE CONFIG
# =======================
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = 'mysql+pymysql://root:soumya%40150905@localhost/fd_rates_db'
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


# =======================
# DATABASE MODEL
# =======================
class FDRate(db.Model):
    __tablename__ = 'fd_rates'
    id = db.Column(db.Integer, primary_key=True)
    bank = db.Column(db.String(100), nullable=False)
    tenure=db.Column(db.String(100),nullable=False)
    generalized_tenure = db.Column(db.String(100))
    min_ten = db.Column(db.Float,nullable=True)
    max_ten = db.Column(db.Float,nullable=True)
    senior_citizen_rate = db.Column(db.Float)
    scrape_date = db.Column(db.DateTime, default=datetime.utcnow)


# =======================
# TENURE PROCESSING FUNCTIONS
# =======================

def categorize_by_days(min_days, max_days):
    """Returns a standard label (e.g. '1 to 3 months') based on day ranges."""
    avg_days = (min_days + max_days) / 2 if min_days and max_days else np.nan
    if pd.isna(avg_days): 
        return "Other"
    if avg_days <= 30:
        return "Less than 1 month"
    elif avg_days <= 90:
        return "1 to 3 months"
    elif avg_days <= 180:
        return "3 to 6 months"
    elif avg_days <= 365:
        return "6 months to 1 year"
    elif avg_days <= 730:
        return "1 to 2 years"
    elif avg_days <= 1095:
        return "2 to 3 years"
    elif avg_days <= 1825:
        return "3 to 5 years"
    else:
        return "Over 5 years"

def map_tenure(tenure_str):
    t = tenure_str.lower().strip()
    if not any(unit in t for unit in ["day", "month", "year", "y"]):
        return "Other"
    
    range_match = re.search(r'(\d+)\s*(?:-|to)\s*(\d+)\s*days?', t)
    if range_match:
        max_days = int(range_match.group(2))
        if max_days <= 30:
            return "Less than 1 month"
        elif max_days <= 90:
            return "1 to 3 months"
        elif max_days <= 180:
            return "3 to 6 months"
        elif max_days <= 365:
            return "6 months to 1 year"
        elif max_days <= 730:
            return "1 to 2 years"
        elif max_days <= 1095:
            return "2 to 3 years"
        elif max_days <= 1825:
            return "3 to 5 years"
        else:
            return "Over 5 years"

    single_day = re.search(r'(\d+)\s*days?', t)
    if single_day:
        days = int(single_day.group(1))
        if days <= 30:
            return "Less than 1 month"
        elif days <= 90:
            return "1 to 3 months"
        elif days <= 180:
            return "3 to 6 months"
        elif days <= 365:
            return "6 months to 1 year"
        elif days <= 730:
            return "1 to 2 years"
        elif days <= 1095:
            return "2 to 3 years"
        elif days <= 1825:
            return "3 to 5 years"
        else:
            return "Over 5 years"

    month_match = re.search(r'(\d+)\s*months?', t)
    if month_match:
        month = int(month_match.group(1))
        if month <= 1:
            return "Less than 1 month"
        elif month <= 3:
            return "1 to 3 months"
        elif month <= 6:
            return "3 to 6 months"
        elif month <= 12:
            return "6 months to 1 year"
        elif month <= 24:
            return "1 to 2 years"
        elif month <= 36:
            return "2 to 3 years"
        elif month <= 60:
            return "3 to 5 years"
        else:
            return "Over 5 years"

    year_match = re.search(r'(\d+)\s*years?', t)
    if year_match:
        y = int(year_match.group(1))
        if y == 1:
            return "1 to 2 years"
        elif y == 2:
            return "2 to 3 years"
        elif 3 <= y <= 5:
            return "3 to 5 years"
        else:
            return "Over 5 years"

    return "Other"


def convert_to_months(text):
    """Convert tenure text like '1 year' or '6 months' to months."""
    text = text.strip().lower()
    if 'year' in text:
        return float(text.split()[0]) * 12
    elif 'month' in text:
        return float(text.split()[0])
    else:
        return np.nan




import re
import numpy as np

def extract_tenure_range(tenure_text):
    text = str(tenure_text).lower().strip()
    
    # clean out non-tenure words/numbers (like years, dates, IDs)
    text = re.sub(r'[^a-z0-9\s\.\-<>]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    def to_days(value, unit):
        value = float(value)
        if "year" in unit or unit == "y":
            return value * 365
        elif "month" in unit:
            return value * 30
        elif "day" in unit:
            return value
        return np.nan

    # Capture ranges like '4 to 45 days', '6 months to 1 year'
    range_match = re.search(r'(\d+\.?\d*)\s*(day|month|year)?\s*(?:to|-|–|upto|up to)\s*(\d+\.?\d*)\s*(day|month|year)', text)
    if range_match:
        min_val = to_days(range_match.group(1), range_match.group(2) or range_match.group(4))
        max_val = to_days(range_match.group(3), range_match.group(4))
        return (min_val, max_val)

    # Single value like '270 days'
    single_match = re.search(r'(\d+\.?\d*)\s*(day|month|year)', text)
    if single_match:
        val = to_days(single_match.group(1), single_match.group(2))
        return (val, val)

    return (np.nan, np.nan)



# =======================
# SCRAPER HELPERS
# =======================
def get_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)


# =======================
# BANK SCRAPERS
# =======================
def scrape_icici():
    driver = get_driver()
    url = "https://www.icicibank.com/personal-banking/deposits/fixed-deposit/fd-interest-rates"
    driver.get(url)
    wait = WebDriverWait(driver, 20)
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.hidewp")))
    table = driver.find_element(By.CSS_SELECTOR, "table.hidewp")
    rows = table.find_elements(By.TAG_NAME, "tr")
    
    header_row1 = [h.text.strip() for h in rows[0].find_elements(By.TAG_NAME, "th")]
    header_row2 = [h.text.strip() for h in rows[1].find_elements(By.TAG_NAME, "th")]
    headers = [header_row1[0]] + header_row2[1:]
    
    data = []
    for row in rows[2:]:
        cells = row.find_elements(By.TAG_NAME, "td")
        row_data = [cell.text.strip() for cell in cells]
        if row_data:
            data.append(row_data)
    driver.quit()
    
    df = pd.DataFrame(data, columns=headers)
    df = df[[headers[0], headers[-1]]]
    df['Bank'] = 'ICICI Bank'
    df.columns = ['Tenure', 'Senior Citizen Rate', 'Bank']
    return df


def scrape_kotak():
    driver = get_driver()
    url = "https://www.kotak.com/en/personal-banking/deposits/senior-citizen-fixed-deposit/interest-rate.html"
    driver.get(url)
    time.sleep(8)
    html = driver.page_source
    driver.quit()

    soup = BeautifulSoup(html, 'html.parser')
    tbody = soup.find('tbody')
    rows = tbody.find_all('tr')
    data = []
    for row in rows:
        cols = row.find_all('td')
        if len(cols) >= 2:
            maturity = cols[0].text.strip()
            rate = cols[1].text.strip()
            data.append([maturity, rate])
    df = pd.DataFrame(data, columns=["Tenure", "Senior Citizen Rate"])
    df['Bank'] = 'Kotak Mahindra Bank'
    return df


def scrape_sbi():
    driver = get_driver()
    url = "https://sbi.bank.in/web/interest-rates/deposit-rates/retail-domestic-term-deposits"
    driver.get(url)
    time.sleep(8)
    html = driver.page_source
    driver.quit()

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", {"class": "table table-bordered"})
    rows = table.find_all("tr")[2:]  # Skip header
    data = []
    for row in rows:
        cols = row.find_all("td")
        if cols:
            tenor = cols[0].get_text(strip=True)
            rate = cols[4].get_text(strip=True)
            data.append([tenor, rate])
    df = pd.DataFrame(data, columns=["Tenure", "Senior Citizen Rate"])
    df['Bank'] = 'SBI'
    return df


def scrape_union_bank():
    driver = get_driver()
    url = "https://www.etmoney.com/fixed-deposit/union-bank-fd-rates/14"
    driver.get(url)
    time.sleep(8)
    html = driver.page_source
    driver.quit()

    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table', id='t01')
    data = []
    if table:
        rows = table.find_all('tr')[1:]
        for row in rows:
            cols = row.find_all('td')
            if len(cols) == 3:
                tenure = cols[0].get_text(strip=True)
                senior_rate = cols[2].get_text(strip=True)
                data.append([tenure, senior_rate])
    df = pd.DataFrame(data, columns=["Tenure", "Senior Citizen Rate"])
    df['Bank'] = 'Union Bank'
    return df


# =======================
# INSERT INTO DATABASE
# =======================
def insert_fd_rates_sqlalchemy(df):
    with app.app_context():
        for _, row in df.iterrows():
            rate_str = str(row['Senior Citizen Rate']).replace('%', '').strip()
            try:
                rate = float(rate_str)
            except:
                rate = None

            record = FDRate(
                bank=row['Bank'],
                tenure=row['Tenure'],
                generalized_tenure=row['Generalized Tenure'],
                min_ten=row['Min_Tenure'],
                max_ten=row['Max_Tenure'],
                senior_citizen_rate=rate
            )
            db.session.add(record)
        db.session.commit()
    print(f"{len(df)} records inserted successfully.")


# =======================
# MAIN EXECUTION
# =======================
if __name__ == "__main__":
    with app.app_context():
        db.create_all()

        icici_df = scrape_icici()
        kotak_df = scrape_kotak()
        sbi_df = scrape_sbi()
        union_df = scrape_union_bank()

        combined_df = pd.concat([icici_df, kotak_df, sbi_df, union_df], ignore_index=True)
       
        combined_df[['Min_Tenure', 'Max_Tenure']] = combined_df['Tenure'].apply(
                    lambda x: pd.Series(extract_tenure_range(x))
        )
        combined_df['Generalized Tenure'] = combined_df.apply(
            lambda r: categorize_by_days(r['Min_Tenure'], r['Max_Tenure']), axis=1
        )
        combined_df = combined_df.replace({np.nan: None})

        combined_df.to_csv("senior_citizen_fd_rates.csv", index=False)
        insert_fd_rates_sqlalchemy(combined_df)
        print("✅ Scraping, saving to CSV, and database insertion complete!")
