import pandas as pd
import numpy as np
import re

# ============================================
# LOAD CSV
# ============================================

df = pd.read_csv("app/data/senior_citizen_fd_rates.csv")

print("Original Data:")
print(df.head())

# ============================================
# STANDARDIZE COLUMN NAMES
# ============================================

df.columns = [
    col.strip()
    .lower()
    .replace(" ", "_")
    for col in df.columns
]

print("\nStandardized Columns:")
print(df.columns)

# ============================================
# CLEAN INTEREST RATES
# ============================================

def clean_rate(rate):

    try:
        return float(
            str(rate)
            .replace("%", "")
            .replace("*", "")
            .strip()
        )

    except:
        return np.nan


df["senior_citizen_rate"] = (
    df["senior_citizen_rate"]
    .apply(clean_rate)
)

# ============================================
# STANDARDIZE BANK NAMES
# ============================================

df["bank"] = (
    df["bank"]
    .astype(str)
    .str.strip()
    .str.upper()
)

# ============================================
# CONVERT UNIT TO DAYS
# ============================================

def to_days(value, unit):

    value = float(value)

    unit = unit.lower()

    if "year" in unit:
        return int(value * 365)

    elif "month" in unit:
        return int(value * 30)

    elif "day" in unit:
        return int(value)

    return np.nan


# ============================================
# CATEGORY MAPPING
# ============================================

def categorize_by_days(min_days, max_days):

    if pd.isna(min_days) or pd.isna(max_days):
        return "Other"

    avg_days = (min_days + max_days) / 2

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


# ============================================
# TENURE RANGE EXTRACTION
# ============================================

def extract_min_max_tenure(text):

    text = str(text).lower().strip()

    # remove brackets
    text = re.sub(r"\(.*?\)", "", text)

    # normalize spaces
    text = re.sub(r"\s+", " ", text)

    # ====================================================
    # CASE 1:
    # 2 Years 1 Day to 3 Years
    # ====================================================

    one_day_match = re.search(
        r'(\d+)\s*years?\s*1\s*day\s*to\s*(\d+)\s*years?',
        text
    )

    if one_day_match:

        min_days = (
            int(one_day_match.group(1)) * 365
        ) + 1

        max_days = (
            int(one_day_match.group(2)) * 365
        )

        return (min_days, max_days)

    # ====================================================
    # CASE 2:
    # 7 - 14 Days
    # 31-45 Days
    # 46 -90 Days
    # ====================================================

    dash_match = re.search(
        r'(\d+)\s*-\s*(\d+)\s*(day|days|month|months|year|years)',
        text
    )

    if dash_match:

        min_days = to_days(
            dash_match.group(1),
            dash_match.group(3)
        )

        max_days = to_days(
            dash_match.group(2),
            dash_match.group(3)
        )

        return (min_days, max_days)

    # ====================================================
    # CASE 3:
    # 7 days to 45 days
    # 181 Days to 269 Days
    # ====================================================

    range_match = re.search(
        r'(\d+)\s*(day|days|month|months|year|years)\s*to\s*(\d+)\s*(day|days|month|months|year|years)',
        text
    )

    if range_match:

        min_days = to_days(
            range_match.group(1),
            range_match.group(2)
        )

        max_days = to_days(
            range_match.group(3),
            range_match.group(4)
        )

        return (min_days, max_days)

    # ====================================================
    # CASE 4:
    # 1 year to less than 18 months
    # ====================================================

    less_than_match = re.search(
        r'(\d+)\s*(year|years|month|months|day|days)\s*to\s*less than\s*(\d+)\s*(year|years|month|months|day|days)',
        text
    )

    if less_than_match:

        min_days = to_days(
            less_than_match.group(1),
            less_than_match.group(2)
        )

        max_days = (
            to_days(
                less_than_match.group(3),
                less_than_match.group(4)
            ) - 1
        )

        return (min_days, max_days)

    # ====================================================
    # CASE 5:
    # 3 years and above but less than 4 years
    # ====================================================

    above_match = re.search(
        r'(\d+)\s*(year|years|month|months|day|days)\s*and above but less than\s*(\d+)\s*(year|years|month|months|day|days)',
        text
    )

    if above_match:

        min_days = to_days(
            above_match.group(1),
            above_match.group(2)
        )

        max_days = (
            to_days(
                above_match.group(3),
                above_match.group(4)
            ) - 1
        )

        return (min_days, max_days)

    # ====================================================
    # CASE 6:
    # 5 years and above up to and inclusive of 10 years
    # ====================================================

    inclusive_match = re.search(
        r'(\d+)\s*(year|years|month|months|day|days).*?(\d+)\s*(year|years|month|months|day|days)',
        text
    )

    if (
        inclusive_match
        and "inclusive" in text
    ):

        min_days = to_days(
            inclusive_match.group(1),
            inclusive_match.group(2)
        )

        max_days = to_days(
            inclusive_match.group(3),
            inclusive_match.group(4)
        )

        return (min_days, max_days)

    # ====================================================
    # CASE 7:
    # > 1 Year to 399 days
    # ====================================================

    greater_match = re.search(
        r'>\s*(\d+)\s*(year|years|month|months|day|days)\s*to\s*(\d+)\s*(year|years|month|months|day|days)',
        text
    )

    if greater_match:

        min_days = (
            to_days(
                greater_match.group(1),
                greater_match.group(2)
            ) + 1
        )

        max_days = to_days(
            greater_match.group(3),
            greater_match.group(4)
        )

        return (min_days, max_days)

    # ====================================================
    # CASE 8:
    # 400 days
    # 23 Months
    # ====================================================

    single_match = re.search(
        r'^(\d+)\s*(day|days|month|months|year|years)$',
        text
    )

    if single_match:

        value = to_days(
            single_match.group(1),
            single_match.group(2)
        )

        return (value, value)

    # ====================================================
    # FALLBACK
    # ====================================================

    return (np.nan, np.nan)


# ============================================
# APPLY TENURE EXTRACTION
# ============================================

df[["clean_min_days", "clean_max_days"]] = (
    df["tenure"]
    .apply(extract_min_max_tenure)
    .apply(pd.Series)
)

# ============================================
# GENERALIZED TENURE
# ============================================

df["generalized_tenure"] = df.apply(
    lambda row: categorize_by_days(
        row["clean_min_days"],
        row["clean_max_days"]
    ),
    axis=1
)



# ============================================
# REMOVE DUPLICATES
# ============================================

df = df.drop_duplicates()

# ============================================
# NULL CHECK
# ============================================

print("\nMissing Values:")
print(df.isnull().sum())

# ============================================
# REMOVE NULLS
# ============================================

df = df.dropna()
df=df.drop(columns=['min_tenure','max_tenure'])



# ============================================
# SAVE CLEANED DATA
# ============================================

output_path = "app/data/clean_fd_rates.csv"

df.to_csv(
    output_path,
    index=False
)

print(f"\nCleaned data saved to: {output_path}")

# ============================================
# FINAL DATAFRAME INFO
# ============================================

print("\nFinal Data Info:")

print(df.info())