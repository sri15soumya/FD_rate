import pandas as pd

from app.scrapers.icici_scraper import ICICIScraper
from app.scrapers.kotak_scraper import KotakScraper
from app.scrapers.sbi_scraper import SBIScraper
from app.scrapers.union_scraper import UnionBankScraper
from app.scrapers.hdfc_scraper import  HDFCScraper

from app.processors.tenure_processor import (
    extract_tenure_range,
    categorize_by_days
)

from app.database.repository import save_fd_rates

from app.utils.logger import logger


def run_scrapers():

    logger.info("Starting all scrapers")

    scrapers = [

         #HDFCScraper(),
        ICICIScraper(),

        KotakScraper(),

        SBIScraper(),

        UnionBankScraper()
    ]

    dfs = []

    for scraper in scrapers:

        try:

            df = scraper.scrape()

            if not df.empty:
                dfs.append(df)

        except Exception as e:

            logger.error(str(e))

    combined_df = pd.concat(
        dfs,
        ignore_index=True
    )

    combined_df[
        ["Min_Tenure", "Max_Tenure"]
    ] = combined_df["Tenure"].apply(
        lambda x: pd.Series(
            extract_tenure_range(x)
        )
    )

    combined_df["Generalized Tenure"] = combined_df.apply(
        lambda r: categorize_by_days(
            r["Min_Tenure"],
            r["Max_Tenure"]
        ),
        axis=1
    )

    combined_df.to_csv(
        "app/data/senior_citizen_fd_rates.csv",
        index=False
    )

    save_fd_rates(combined_df)

    logger.info("Scraping completed successfully")