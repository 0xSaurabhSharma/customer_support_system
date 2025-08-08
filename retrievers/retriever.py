import os
# from langchain_astradb import AstraDBVectorStore
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec, Pinecone
from typing import List
from langchain_core.documents import Document
from utils.config_loader import load_config
from utils.model_loader import ModelLoader
from dotenv import load_dotenv

class Retriever:
    
    def __init__(self):
        self.model_loader=ModelLoader()
        self.config=load_config()
        self._load_env_variables()
        self.vstore = None
        self.retriever = None
    
    def _load_env_variables(self):
         
        load_dotenv()
         
        required_vars = ["GOOGLE_API_KEY", "PINECONE_API_KEY"
                        #  "ASTRA_DB_API_ENDPOINT", "ASTRA_DB_APPLICATION_TOKEN"
                        #  , "ASTRA_DB_KEYSPACE"
                         ]
        
        missing_vars = [var for var in required_vars if os.getenv(var) is None]
        
        if missing_vars:
            raise EnvironmentError(f"Missing environment variables: {missing_vars}")

        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        # self.db_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
        # self.db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        # self.db_keyspace = os.getenv("ASTRA_DB_KEYSPACE")
        
    
    def load_retriever(self):
        if not self.vstore:
            
            # collection_name = self.config["astra_db"]["collection_name"]
            # vstore = AstraDBVectorStore(
            #     embedding= self.model_loader.load_embeddings(),
            #     api_endpoint=self.db_api_endpoint,
            #     token=self.db_application_token,
            #     collection_name=collection_name,
            #     namespace="default_keyspace",  
            # )      
            
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            pc = Pinecone(api_key=pinecone_api_key)
            
            self.vstore = PineconeVectorStore(
                index=pc.Index(self.config["pinecone_db"]["index_name"]), 
                embedding= self.model_loader.load_embeddings(),
            )
            
        
        if not self.retriever:
            
            top_k = self.config["retriever"]["top_k"] if "retriever" in self.config else 3
            retriever = self.vstore.as_retriever(search_kwargs={"k": top_k})
            
            retriever = self.vstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": top_k,
                    "score_threshold": 0.5 #self.config["retriever"]["score_threshold"]
                }
            )
            
            print("Retriever loaded successfully.")
            return retriever
   

    
    def call_retriever(self,query:str)-> List[Document]:
        retriever=self.load_retriever()
        output=retriever.invoke(query)
        return output
        
    
if __name__=='__main__':
    retriever_obj = Retriever()
    user_query = "Can you suggest good budget laptops?"
    results = retriever_obj.call_retriever(user_query)

    for idx, doc in enumerate(results, 1):
        print(f"Result {idx}: {doc.page_content}\nMetadata: {doc.metadata}\n")