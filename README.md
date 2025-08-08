# 🛍️ Customer Support Q&A System

An AI-powered chatbot that ingests product reviews and metadata, stores embeddings in AstraDB/Pinecone, and answers user queries via a FastAPI backend and a simple HTML/JS frontend. It integrates Google embeddings, Groq LLM, and vector search for real-time recommendation and support.

---

## 🚀 Features

- **Data Ingestion**  
  • Reads CSV of Flipkart product reviews  
  • Transforms into LangChain `Document` objects  

- **Vector Store**  
  • Uses Pinecone or AstraDB for embedding storage  
  • Configurable namespace/collection  

- **Embedding & LLM**  
  • Google `text-embedding-004` embeddings (dim=768)  
  • Groq or Google Gemini for answer generation  

- **Backend**  
  • FastAPI app (`main.py`) with `/get` endpoint  
  • CORS-enabled, async endpoints  

- **Frontend**  
  • Simple HTML/JS (`templates/chat.html`)  
  • Tailored CSS (`static/styles.css`)  

- **Config & Secrets**  
  • `.env` for API keys & endpoints  
  • `config/config.yml` for thresholds & parameters  

---

## 📂 Project Structure

```

.
├── .env                      # API keys & endpoints
├── main.py                   # FastAPI entrypoint
├── requirements.txt
├── setup.py
├── README.md
│
├── config/
│   └── config.yml            # Retriever & pipeline settings
│
├── data/
│   └── flipkart\_product\_review\.csv
│
├── data\_ingestion/
│   └── data\_ingest.py        # Ingestion pipeline class
│
├── exceptions/
│   └── exception.py          # Custom exception classes
│
├── logs/
│   └── logging.py            # Logger setup
│
├── models/
│   └── model.py              # Pydantic schemas
│
├── prompts/
│   └── prompt.py             # Prompt templates
│
├── retrievers/
│   └── retriever.py          # Vector store & retriever logic
│
├── static/
│   └── styles.css
│
├── templates/
│   └── chat.html
│
└── utils/
├── config\_loader.py      # Loads `config.yml`
└── model\_loader.py       # Loads embedding & LLM models

````

---

## ⚙️ Setup & Installation

1. **Clone & enter project**  
   ```bash
   git clone https://github.com/yourname/customer_support_system.git
   cd customer_support_system
   ```

2. **Create & activate virtual environment**

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure `.env`**
   Create a `.env` file in the root with:

   ```env
   GOOGLE_API_KEY=your_google_key
   GROQ_API_KEY=your_groq_key
   PINECONE_API_KEY=your_pinecone_key
   PINECONE_ENVIRONMENT=your_pinecone_env
   ASTRA_DB_API_ENDPOINT=your_astra_endpoint
   ASTRA_DB_APPLICATION_TOKEN=your_astra_token
   ASTRA_DB_KEYSPACE=your_keyspace
   ```

5. **Edit `config/config.yml`**
   Define your retriever and ingestion settings:

   ```yaml
   retriever:
     top_k: 5
     score_threshold: 0.1

   ingestion:
     batch_size: 100
     collection_name: customer_reviews
   ```

---

## 🚦 Running the App

1. **Start FastAPI**

   ```bash
   uvicorn main:app --reload --port 8000
   ```

2. **Open Frontend**
   Navigate to `http://localhost:8000` in your browser.

3. **Interact**
   • Ask product-related questions
   • Watch backend logs for ingestion & retrieval details

---

## 🔧 Customization

* **Swap vector store**: in `retriever.py`, switch between `AstraDBVectorStore` and `Pinecone`
* **Change embedding model**: in `model_loader.py`, pick another Google or OpenAI model
* **Extend prompts**: edit `prompts/prompt.py` for different Q\&A personalities

---

## 🤝 Contributing

1. Fork this repo
2. Create a feature branch (`git checkout -b feat/your-feature`)
3. Commit & push (`git commit -m "Add feature"` + `git push`)
4. Open a PR for review

---

## 📄 License

MIT © \[Your Name]
Feel free to learn, modify, and share!
