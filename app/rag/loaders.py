import pandas as pd

from langchain_core.documents import Document

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader
)


# ====================================================
# LOAD FD CSV AS DOCUMENTS
# ====================================================

def load_fd_rate_documents():

    df = pd.read_csv(
        "app/data/clean_fd_rates.csv"
    )

    documents = []

    for _, row in df.iterrows():

        text = f"""
        Bank: {row['bank']}

        Senior Citizen FD Rate:
        {row['senior_citizen_rate']}%

        Tenure:
        {row['tenure']}

        Minimum Days:
        {row['clean_min_days']}

        Maximum Days:
        {row['clean_max_days']}

        Generalized Tenure:
        {row['generalized_tenure']}

        """

        doc = Document(
            page_content=text,
            metadata={
                "source": "fd_rates",
                "bank": row["bank"]
            }
        )

        documents.append(doc)

    return documents


# ====================================================
# LOAD TERMS & CONDITIONS
# ====================================================

def load_term_documents():

    loader = DirectoryLoader(
        "app/data/terms",
        glob="*.txt",
        loader_cls=TextLoader
    )

    documents = loader.load()

    return documents


# ====================================================
# LOAD ALL DOCUMENTS
# ====================================================

def load_all_documents():

    fd_documents = load_fd_rate_documents()

    term_documents = load_term_documents()

    all_documents = (
        fd_documents + term_documents
    )

    print(
        f"Loaded {len(fd_documents)} FD docs"
    )

    print(
        f"Loaded {len(term_documents)} T&C docs"
    )

    print(
        f"Total documents: {len(all_documents)}"
    )

    return all_documents