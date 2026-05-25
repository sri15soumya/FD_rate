from app.rag.loaders import (
    load_all_documents
)

from app.rag.chunker import (
    split_documents
)

from app.rag.vectorstore import (
    create_vectorstore
)


def ingest_pipeline():

    documents = load_all_documents()

    chunks = split_documents(
        documents
    )

    print(
        f"Created {len(chunks)} chunks"
    )

    create_vectorstore(chunks)

    print(
        "Vector DB created successfully"
    )


if __name__ == "__main__":

    ingest_pipeline()