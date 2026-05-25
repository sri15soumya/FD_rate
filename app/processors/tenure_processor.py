import re
import numpy as np
import pandas as pd


def categorize_by_days(min_days, max_days):

    avg_days = (
        (min_days + max_days) / 2
        if min_days and max_days
        else np.nan
    )

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

    return "Over 5 years"


# =======================
# TENURE MAPPING
# =======================

def map_tenure(tenure_str):

    t = tenure_str.lower().strip()

    if not any(
        unit in t
        for unit in ["day", "month", "year", "y"]
    ):
        return "Other"

    range_match = re.search(
        r'(\d+)\s*(?:-|to)\s*(\d+)\s*days?',
        t
    )

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

        return "Over 5 years"

    single_day = re.search(
        r'(\d+)\s*days?',
        t
    )

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

        return "Over 5 years"

    month_match = re.search(
        r'(\d+)\s*months?',
        t
    )

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

        return "Over 5 years"

    year_match = re.search(
        r'(\d+)\s*years?',
        t
    )

    if year_match:

        y = int(year_match.group(1))

        if y == 1:
            return "1 to 2 years"

        elif y == 2:
            return "2 to 3 years"

        elif 3 <= y <= 5:
            return "3 to 5 years"

        return "Over 5 years"

    return "Other"


# =======================
# MONTH CONVERSION
# =======================

def convert_to_months(text):

    text = text.strip().lower()

    if 'year' in text:
        return float(text.split()[0]) * 12

    elif 'month' in text:
        return float(text.split()[0])

    return np.nan


# =======================
# TENURE RANGE EXTRACTION
# =======================

def extract_tenure_range(tenure_text):

    text = str(tenure_text).lower().strip()

    text = re.sub(
        r'[^a-z0-9\s\.\-<>]',
        ' ',
        text
    )

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

    range_match = re.search(
        r'(\d+\.?\d*)\s*(day|month|year)?\s*(?:to|-|–|upto|up to)\s*(\d+\.?\d*)\s*(day|month|year)',
        text
    )

    if range_match:

        min_val = to_days(
            range_match.group(1),
            range_match.group(2) or range_match.group(4)
        )

        max_val = to_days(
            range_match.group(3),
            range_match.group(4)
        )

        return (min_val, max_val)

    single_match = re.search(
        r'(\d+\.?\d*)\s*(day|month|year)',
        text
    )

    if single_match:

        val = to_days(
            single_match.group(1),
            single_match.group(2)
        )

        return (val, val)

    return (np.nan, np.nan)