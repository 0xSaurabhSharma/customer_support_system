import os
# from langchain_astradb import AstraDBVectorStore
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec, Pinecone
from typing import List
from langchain_core.documents import Document
from utils.config_loader import load_config
from utils.model_loader import ModelLoader
from utils.settings_loader import settings

class Retriever:
    
    def __init__(self):
        self.settings = settings
        self.model_loader=ModelLoader()
        self.config=load_config()
        self.vstore = None
        self.retriever = None
        
    
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
            pc = Pinecone(api_key=self.settings.PINECONE_API_KEY)
            
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