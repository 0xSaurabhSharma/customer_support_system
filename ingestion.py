if __name__ == "__main__":
    # 1️⃣ Test Model Loader
    from utils.model_loader import ModelLoader
    loader = ModelLoader()
    embeddings = loader.load_embeddings()
    print("✅ Embedding model loaded successfully:", embeddings)

    # 2️⃣ Test Data Ingestion (CSV → Pinecone)
    
    from data_ingestion.data_ingest import DataIngestion
    ingestion = DataIngestion()
    docs = ingestion.transform_data()
    print(f"✅ Transformed {len(docs)} documents.")

    vstore, ids = ingestion.store_in_vector_db(docs)
    print(f"✅ Stored {len(ids)} documents in vector DB.")

    # 3️⃣ Test Retriever
    retriever = vstore.as_retriever(search_kwargs={"k": 3})
    query = "low budget headphone"
    results = retriever.get_relevant_documents(query)

    print(f"🔍 Query: {query}")
    for r in results:
        print(f"- {r.page_content[:60]}... | Metadata: {r.metadata}")
