import os
from services.data_processing_service import process_files, store_in_pinecone

DATA_DIR = "data"

if __name__ == "__main__":
    # Collect all PDF and CSV files in data folder
    files_to_process = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.lower().endswith((".pdf", ".csv"))
    ]

    # Process and store
    chunks = process_files(files_to_process)
    # store_in_pinecone(chunks)
