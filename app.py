import os
import streamlit as st
import time
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_cohere import ChatCohere
from langchain.chains import RetrievalQA

# Import ONLY the Config class
from config import Config

# ==================== PAGE CONFIGURATION ====================
load_dotenv()
st.set_page_config(
    page_title=Config.APP_TITLE,
    page_icon=Config.APP_ICON,
    layout=Config.PAGE_LAYOUT,
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
/* Main styling */
.stChatMessage {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 0.5rem;
}
/* User message styling */
.stChatMessage[data-testid="user-message"] {
    background-color: #e3f2fd;
}
/* Assistant message styling */
.stChatMessage[data-testid="assistant-message"] {
    background-color: #f5f5f5;
}
/* Sample question buttons */
.stButton>button {
    width: 100%;
    text-align: left;
    padding: 0.5rem 1rem;
    border-radius: 0.5rem;
}
/* Center the logo */
div[data-testid="stImage"] {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ==================== HELPER FUNCTIONS ====================

@st.cache_resource
def load_rag_pipeline():
    """Loads the vector store, LLM, and creates the RAG chain."""
    
    # Load Vector Store
    if not os.path.exists(Config.VECTOR_STORE_PATH):
        st.error(f"Vector store not found. Please run 'python ingest.py' first.")
        st.stop()
        
    embeddings = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    db = FAISS.load_local(str(Config.VECTOR_STORE_PATH), embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={'k': Config.RETRIEVAL_TOP_K})

    # Load LLM (Cohere)
    if not Config.COHERE_API_KEY:
        st.error("COHERE_API_KEY not found. Please add it to your .env file.")
        st.stop()
        
    llm = ChatCohere(
        model=Config.LLM_MODEL_NAME,
        cohere_api_key=Config.COHERE_API_KEY,
        temperature=Config.LLM_TEMPERATURE
    )

    # Create Prompt Template
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=Config.SYSTEM_PROMPT
    )

    # Create the RAG Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )
    return qa_chain

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": Config.WELCOME_MESSAGE,
            "timestamp": datetime.now()
        }]
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0
    if "total_latency" not in st.session_state:
        st.session_state.total_latency = 0.0

def add_message(role: str, content: str):
    """Add a message to chat history."""
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now()
    })
    if len(st.session_state.messages) > Config.MAX_CHAT_HISTORY:
        st.session_state.messages = st.session_state.messages[-Config.MAX_CHAT_HISTORY:]

def format_source_display() -> str:
    """
    Creates a simple, clean "Source" line.
    """
    return "\n\n---\n**📚 Source:** GUVI Knowledge Base\n"

# ==================== SIDEBAR ====================

def render_sidebar():
    """Render the sidebar with info and controls"""
    with st.sidebar:
        st.title("🎓 " + Config.APP_TITLE)
        st.markdown("---")

        # About section
        st.subheader("About This RAG Model")
        st.info(
            """
            This AI RAG model answers questions about GUVI's:

            ✅ **Courses** - Full Stack, Data Science, AI/ML

            ✅ **Zen Class** - Premium program

            ✅ **Policies** - Refunds, terms, support

            ✅ **Platform** - Features and learning tools
            
            """
        )
        st.markdown("---")

        # Sample questions
        st.subheader("💡 Try These Questions")
        for question in Config.SAMPLE_QUESTIONS:
            if st.button(question, key=f"sample_{question}", use_container_width=True):
                st.session_state.sample_question = question
                st.rerun()
        st.markdown("---")

        # Performance metrics
        st.subheader("📊 Performance Metrics")
        total_queries = st.session_state.total_queries
        total_latency = st.session_state.total_latency
        avg_latency = (total_latency / total_queries) if total_queries > 0 else 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Queries", total_queries)
        with col2:
            st.metric("Avg Latency", f"{avg_latency:.2f}s")
        st.markdown("---")

        # Controls
        st.subheader("🛠️ Controls")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = [{
                "role": "assistant",
                "content": Config.WELCOME_MESSAGE,
                "timestamp": datetime.now()
            }]
            st.session_state.total_queries = 0
            st.session_state.total_latency = 0.0
            st.rerun()

        # Chat export
        chat_export = "\n\n".join([
            f"[{msg['timestamp'].strftime('%H:%M:%S')}] {msg['role'].upper()}: {msg['content']}"
            for msg in st.session_state.messages
        ])
        st.download_button(
            label="📥 Export Chat",
            data=chat_export,
            file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
        st.markdown("---")

        # Footer
        st.caption("Made by HIDA for GUVI Capstone Project")
        st.caption("Powered by RAG, Cohere, & Streamlit")

# ==================== MAIN APP ====================

def main():
    """Main application function"""
    
    initialize_session_state()
    qa_chain = load_rag_pipeline()
    
    # --- Centered Logo ---
    # Make sure you have 'logo.png' in the same folder as app.py
    if os.path.exists("logo.png"):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image("logo.png", width=400) # Adjust width as needed
    
    render_sidebar()
    
    st.title(Config.APP_TITLE)
    st.caption("Your RAG-powered guide to GUVI's courses, programs, and policies")

    # --- Handle Sample Question Click ---
    if "sample_question" in st.session_state:
        user_query = st.session_state.sample_question
        del st.session_state.sample_question
        
        add_message("user", user_query)
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                start_time = time.time()
                response = qa_chain.invoke({"query": user_query})
                latency = time.time() - start_time
                
                # --- FIX: Changed 'answer' to 'result' ---
                answer = response['result'] 
                answer += format_source_display()
                answer += f"\n\n*Response generated in {latency:.2f}s*"
                
                st.markdown(answer)
                
                add_message("assistant", answer)
                st.session_state.total_queries += 1
                st.session_state.total_latency += latency
        st.rerun() # Rerun to clear the sample question state

    # --- Display Chat History ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Chat Input ---
    if prompt := st.chat_input("💬 Ask me anything about GUVI..."):
        add_message("user", prompt)
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching GUVI knowledge base..."):
                start_time = time.time()
                response = qa_chain.invoke({"query": prompt})
                latency = time.time() - start_time
                
                # --- FIX: Changed 'answer' to 'result' ---
                answer = response['result'] 
                answer += format_source_display() # Use the clean source format
                answer += f"\n\n*Response generated in {latency:.2f}s*"
                
                st.markdown(answer)
                
                add_message("assistant", answer)
                st.session_state.total_queries += 1
                st.session_state.total_latency += latency
    
    st.markdown("---")
    st.caption("💡 **Tip:** Be specific in your questions for better answers.")

# ==================== ENTRY POINT ====================

if __name__ == "__main__":
    main()