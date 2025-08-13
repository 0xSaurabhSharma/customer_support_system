import os
import logging
import pandas as pd
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec, Pinecone
from utils.config_loader import load_config
from utils.model_loader import ModelLoader
from utils.settings_loader import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DataIngestion:
    """
    Handles transformation of product review data and ingestion into a Pinecone vector database.
    """

    def __init__(self):
        logging.info("Initializing DataIngestion pipeline...")

        self.settings = settings
        self.model_loader = ModelLoader()
        self.config = load_config()

        base_path = self.settings.CSV_BASE_PATH or os.getcwd()
        self.csv_path = os.path.join(base_path, "products.csv")

        if not os.path.isfile(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.product_data = self._load_csv()

    def _load_csv(self) -> pd.DataFrame:
        """
        Load product data from CSV and validate required columns.
        """
        logging.info(f"Loading CSV data from {self.csv_path}...")
        df = pd.read_csv(self.csv_path)

        required_columns = {"product_title", "rating", "summary", "review"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        return df

    def transform_data(self) -> List[Document]:
        """
        Convert product data into a list of LangChain Document objects.
        """
        logging.info("Transforming CSV data into documents...")
        documents = [
            Document(
                page_content=row["review"],
                metadata={
                    "product_name": row["product_title"],
                    "product_rating": row["rating"],
                    "product_summary": row["summary"],
                },
            )
            for _, row in self.product_data.iterrows()
        ]

        logging.info(f"Transformed {len(documents)} documents.")
        return documents

    def store_in_vector_db(self, documents: List[Document]) -> Tuple[PineconeVectorStore, List[str]]:
        """
        Store documents into Pinecone vector store.
        """
        pinecone_client = Pinecone(api_key=self.settings.PINECONE_API_KEY)
        index_name = self.config["pinecone_db"]["index_name"]

        existing_indexes = [i.name for i in pinecone_client.list_indexes()]
        if index_name not in existing_indexes:
            logging.info(f"Creating Pinecone index: {index_name}")
            pinecone_client.create_index(
                name=index_name,
                dimension=768,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                metric="cosine",
            )

        index = pinecone_client.Index(index_name)
        vstore = PineconeVectorStore(index=index, embedding=self.model_loader.load_embeddings())

        logging.info(f"Adding {len(documents)} documents to Pinecone index...")
        inserted_ids = vstore.add_documents(documents)
        logging.info(f"Successfully inserted {len(inserted_ids)} documents into PineconeDB.")

        return vstore, inserted_ids

    def run_pipeline(self):
        """
        Execute the full pipeline: transform data and store it in the vector DB.
        """
        documents = self.transform_data()
        vstore, _ = self.store_in_vector_db(documents)

        # Test query
        query = "Can you tell me the low budget headphone?"
        results = vstore.similarity_search(query, k=3)

        logging.info(f"\nSample search results for query: '{query}'")
        for res in results:
            logging.info(f"Content: {res.page_content}\nMetadata: {res.metadata}\n")


if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.run_pipeline()
