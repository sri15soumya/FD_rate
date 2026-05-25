from app.database.connection import engine
from app.database.models import Base

from app.services.scrape_service import run_scrapers


Base.metadata.create_all(bind=engine)

run_scrapers()