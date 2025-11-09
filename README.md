# Guvi-RAG-Chatbot 🤖

This project is an intelligent, retrieval-augmented generation (RAG) chatbot built for the GUVI EdTech platform. It provides instant, accurate, and detailed answers to student queries by retrieving information from a private knowledge base of GUVI's course materials, FAQs, and documentation.

This project was built as a capstone for the GUVI Zen Class.


---

## 🚀 Features

* **RAG Pipeline:** Uses a `Retrieval-Augmented Generation` pipeline to ensure answers are factual and based *only* on GUVI's data.
* **Strict Context:** The AI is instructed to *only* answer based on retrieved context and will politely decline to answer off-topic questions.
* **Rich UI:** Built with Streamlit, featuring a real-time chat interface, a professional sidebar, and a centered logo.
* **Chat Controls:** Includes "Clear Chat" and "Export Chat" functionality.
* **Performance Metrics:** The sidebar displays the total number of queries and the average response latency.
* **Modular Code:** The project is cleanly split into `config.py` (for settings), `ingest.py` (for data processing), and `app.py` (for the UI).

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Backend & Orchestration:** LangChain
* **LLM (Generation):** Cohere (via API)
* **Embeddings (Retrieval):** Hugging Face (`sentence-transformers/all-MiniLM-L6-v2`)
* **Vector Database:** FAISS (Facebook AI Similarity Search)

---

## ⚙️ How It Works: The RAG Pipeline

This project is built on two main scripts:

### 1. `ingest.py` (Data Ingestion)
This script is run once to build the knowledge base:
1.  **Load:** Reads the `guvi_data.txt` file from the `/data` folder.
2.  **Chunk:** Splits the large text document into 1000-character chunks.
3.  **Embed:** Uses a Hugging Face model (`all-MiniLM-L6-v2`) to convert each text chunk into a numerical vector.
4.  **Store:** Saves all these vectors into a local, high-speed FAISS database in the `vector_store/` folder.

### 2. `app.py` (The Chat Application)
This script runs the live chatbot:
1.  **Load Pipeline:** Loads the pre-built FAISS database and initializes the Cohere LLM.
2.  **User Query:** A user asks a question (e.g., "What is Zen Class?").
3.  **Retrieve:** The app embeds the user's question and uses FAISS to find the top 5 most relevant text chunks from the `guvi_data.txt` file.
4.  **Augment:** The app "augments" a prompt by stuffing the retrieved chunks (the context) and the user's query into a strict system prompt.
5.  **Generate:** This complete prompt is sent to the Cohere API, which generates a detailed, 4-8 sentence answer based *only* on the provided context.
6.  **Display:** The final answer, source, and latency are shown to the user in the Streamlit UI.

---

## Setup & Installation

Follow these steps to run the project locally.

### 1. Clone the Repository

git clone https://github.com/Hida-Fathima/Guvi-RAG-Chatbot

cd Guvi_RAG_Chatbot


### 2. Create and Activate a Virtual Environment


* Create the environment
 
python -m venv guvi_rag_env

 >Activate on Windows
 
.\guvi_rag_env\Scripts\activate

 >Activate on Mac/Linux
 
source guvi_rag_env/bin/activate

### 3. Install Dependencies


pip install -r requirements.txt

### 4. Set Up Your API Key

* Create a file named .env in the main project folder.

* Paste your Cohere API key into the .env file:

COHERE_API_KEY=your_trial_key_paste_it_here

### 5. Build the Vector Database

You only need to run this command once (or any time you update guvi_data.txt).


python ingest.py

This will create the vector_store/ folder containing your db_faiss.pkl and db_faiss.faiss files.

### 6. Run the Streamlit App


streamlit run app.py
