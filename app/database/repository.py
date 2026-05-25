from app.database.connection import SessionLocal
from app.database.models import FDRate
from app.utils.logger import logger

## logs the data into the data base 

# Receive DataFrame
# Loop through rows
# Clean rate values
# Convert rows into ORM objects
# Store objects in list
# Bulk insert into PostgreSQL
# Commit transaction
# Close session

def save_fd_rates(df):

    session = SessionLocal()

    try:

        records = []

        for _, row in df.iterrows():

            rate = str(row["Senior Citizen Rate"])

            rate = rate.replace("%", "").strip()

            try:
                rate = float(rate)

            except:
                rate = None

            record = FDRate(

                bank=row["Bank"],

                tenure=row["Tenure"],

                generalized_tenure=row["Generalized Tenure"],

                min_ten=row["Min_Tenure"],

                max_ten=row["Max_Tenure"],

                senior_citizen_rate=rate
            )

            records.append(record)

        session.bulk_save_objects(records)

        session.commit()

        logger.info(f"Inserted {len(records)} records")

    except Exception as e:

        session.rollback()

        logger.error(str(e))

    finally:
        session.close()