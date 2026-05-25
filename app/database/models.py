from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class FDRate(Base):

    __tablename__ = "fd_rates"

    id = Column(Integer, primary_key=True, index=True)

    bank = Column(String(100), nullable=False)

    tenure = Column(String(100), nullable=False)

    generalized_tenure = Column(String(100))

    min_ten = Column(Float)

    max_ten = Column(Float)

    senior_citizen_rate = Column(Float)

    scrape_date = Column(DateTime, default=datetime.utcnow)