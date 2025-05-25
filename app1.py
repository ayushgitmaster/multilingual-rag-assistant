import streamlit as st
import whisper
import numpy as np
import tempfile
import google.generativeai as genai
from gtts import gTTS
import os
import soundfile as sf
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from typing import List, Tuple
import base64
import time

# Page config
st.set_page_config(
    page_title="🎤 Voice RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load and cache models"""
    try:
        whisper_model = whisper.load_model("base")
        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        return whisper_model, embedding_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

@st.cache_data
def load_and_process_documents():
    """Load and process documents into FAISS index"""
    try:
        # Sample documents (you can modify this)
        documents = [
            {
                'content': """
                Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines.
                Machine Learning is a subset of AI that enables computers to learn without being explicitly programmed.
                Deep Learning uses neural networks with multiple layers to model and understand complex patterns.
                Natural Language Processing (NLP) helps computers understand and generate human language.
                RAG (Retrieval-Augmented Generation) combines information retrieval with text generation for better responses.
                Voice assistants use speech recognition and text-to-speech technologies.
                Vector databases store high-dimensional vectors for similarity search.
                Embeddings convert text into numerical representations that capture semantic meaning.
                """,
                'source': 'AI Knowledge Base'
            },
        ]
        
        # Split into chunks
        doc_chunks = []
        for i, doc in enumerate(documents):
            chunks = split_text(doc['content'], chunk_size=300, overlap=50)
            for j, chunk in enumerate(chunks):
                doc_chunks.append({
                    'content': chunk.strip(),
                    'source': doc['source'],
                    'chunk_id': f"{i}_{j}"
                })
        
        return doc_chunks
    except Exception as e:
        st.error(f"Error processing documents: {e}")
        return []

def split_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
        
        if i + chunk_size >= len(words):
            break
    
    return chunks

@st.cache_resource
def create_faiss_index(_embedding_model, doc_chunks):
    """Create FAISS index from document chunks"""
    try:
        if not doc_chunks:
            return None, []
            
        chunk_texts = [chunk['content'] for chunk in doc_chunks]
        embeddings = _embedding_model.encode(chunk_texts)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings.astype('float32'))
        
        return index, doc_chunks
    except Exception as e:
        st.error(f"Error creating FAISS index: {e}")
        return None, []

def retrieve_relevant_chunks(query: str, embedding_model, index, doc_chunks, k: int = 3):
    """Retrieve relevant chunks for query"""
    try:
        if index is None or not doc_chunks:
            return []
            
        query_embedding = embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = index.search(query_embedding.astype('float32'), k)
        
        relevant_chunks = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(doc_chunks):
                chunk = doc_chunks[idx].copy()
                chunk['score'] = float(score)
                relevant_chunks.append(chunk)
        
        return relevant_chunks
    except Exception as e:
        st.error(f"Error retrieving chunks: {e}")
        return []

def generate_rag_response(query: str, api_key: str, embedding_model, index, doc_chunks):
    """Generate response using RAG"""
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Retrieve relevant chunks
        relevant_chunks = retrieve_relevant_chunks(query, embedding_model, index, doc_chunks)
        
        if not relevant_chunks:
            return "I couldn't find relevant information to answer your question.", []
        
        # Create context
        context = "\n\n".join([chunk['content'] for chunk in relevant_chunks])
        
        # Generate response
        prompt = f"""
        Based on the following context, please answer the question clearly and concisely.
        If the answer is not in the context, say so.

        Context:
        {context}

        Question: {query}

        Answer:
        """
        
        response = model.generate_content(prompt)
        return response.text, relevant_chunks
        
    except Exception as e:
        return f"Error generating response: {str(e)}", []

def transcribe_audio(audio_bytes, whisper_model):
    """Transcribe audio bytes to text"""
    try:
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        # Transcribe
        result = whisper_model.transcribe(tmp_file_path)
        
        # Clean up
        os.unlink(tmp_file_path)
        
        return result["text"].strip(), result["language"]
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return None, None

import pyttsx3
import tempfile
import os

def text_to_speech(text: str, lang: str = "en"):
    """Convert text to speech and return audio bytes"""
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # Adjust speech rate if desired

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file_path = tmp_file.name

        engine.save_to_file(text, tmp_file_path)
        engine.runAndWait()

        with open(tmp_file_path, "rb") as audio_file:
            audio_bytes = audio_file.read()

        os.unlink(tmp_file_path)
        return audio_bytes
    except Exception as e:
        st.error(f"Error in text-to-speech: {e}")
        return None

def create_audio_player(audio_bytes):
    """Create HTML audio player"""
    if audio_bytes:
        audio_base64 = base64.b64encode(audio_bytes).decode()
        audio_html = f"""
        <audio controls autoplay>
            <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
            Your browser does not support the audio element.
        </audio>
        """
        return audio_html
    return ""

# 

def main():
    # Header
    st.markdown('<h1 class="main-header">🎤 Multilingual RAG Assistant</h1>', unsafe_allow_html=True)
    st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .success-box {
        background-color: #1e1e1e;  /* Dark background */
        color: #ffffff;             /* White text */
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "🔑 Gemini API Key",
            type="password",
            help="Get your API key from Google AI Studio"
        )
        
        if not api_key:
            st.markdown('<div class="info-box">📝 Please enter your Gemini API key to continue</div>', 
                       unsafe_allow_html=True)
            st.markdown("[Get API Key](https://makersuite.google.com/app/apikey)")
            return
        
        st.success("✅ API Key configured!")
        
        # Model loading
        st.header("🤖 Model Status")
        with st.spinner("Loading models..."):
            whisper_model, embedding_model = load_models()
            
        if whisper_model is None or embedding_model is None:
            st.error("❌ Failed to load models")
            return
        
        st.success("✅ Models loaded!")
        
        # Document processing
        st.header("📚 Knowledge Base")
        with st.spinner("Processing documents..."):
            doc_chunks = load_and_process_documents()
            index, doc_chunks = create_faiss_index(embedding_model, doc_chunks)
        
        if index is None:
            st.error("❌ Failed to create knowledge base")
            return
            
        st.success(f"✅ {len(doc_chunks)} chunks indexed!")
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["🎤 Voice Query", "💬 Text Query", "📖 About"])
    
    with tab1:
        st.header("🎤 Voice Query")
        st.markdown("Upload your recorded question (wav/mp3) and get an AI-powered response!")
        
        # File uploader instead of experimental_audio_input
        audio_file = st.file_uploader("Upload your audio question:", type=["wav", "mp3"])
        
        if audio_file is not None:
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format=f"audio/{audio_file.type.split('/')[-1]}")
            
            if st.button("🔄 Process Voice Query", type="primary"):
                with st.spinner("Processing your voice query..."):
                    # Transcribe
                    query_text, language = transcribe_audio(audio_bytes, whisper_model)
                    
                    if query_text:
                        st.markdown(f"**📝 Your Question:** {query_text}")
                        st.markdown(f"**🌐 Detected Language:** {language}")
                        
                        # Generate response
                        response, sources = generate_rag_response(
                            query_text, api_key, embedding_model, index, doc_chunks
                        )
                        
                        # Display response
                        st.markdown("**🤖 AI Response:**")
                        st.markdown(f'<div class="success-box">{response}</div>', 
                                   unsafe_allow_html=True)
                        
                        # Show sources
                        if sources:
                            st.markdown("**📚 Sources:**")
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**{i}.** {source['source']} (Relevance: {source.get('score', 0):.3f})")
                        
                        # Text-to-speech
                        st.markdown("**🔊 Audio Response:**")
                        audio_response = text_to_speech(response, language)
                        if audio_response:
                            audio_html = create_audio_player(audio_response)
                            st.markdown(audio_html, unsafe_allow_html=True)
                    else:
                        st.error("❌ Could not transcribe audio. Please try again.")
    
    with tab2:
        st.header("💬 Text Query")
        st.markdown("Type your question for testing without voice input.")
        
        query = st.text_input("Enter your question:", placeholder="What is artificial intelligence?")
        
        if query and st.button("🔍 Get Answer", type="primary"):
            with st.spinner("Generating response..."):
                response, sources = generate_rag_response(
                    query, api_key, embedding_model, index, doc_chunks
                )
                
                st.markdown("**🤖 AI Response:**")
                st.markdown(f'<div class="success-box" bgcolor="blue">{response}</div>', 
                           unsafe_allow_html=True)
                
                if sources:
                    st.markdown("**📚 Sources:**")
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**{i}.** {source['source']} (Relevance: {source.get('score', 0):.3f})")
                
                # Audio response
                audio_response = text_to_speech(response, "en")
                if audio_response:
                    st.markdown("**🔊 Audio Response:**")
                    audio_html = create_audio_player(audio_response)
                    st.markdown(audio_html, unsafe_allow_html=True)
    
    with tab3:
        st.header("📖 About Voice RAG Assistant")
        st.markdown("""
        ### 🚀 Features
        - **Voice Input**: Upload audio files (wav/mp3) with your question
        - **Speech Recognition**: Powered by OpenAI Whisper
        - **Smart Retrieval**: Uses FAISS for fast similarity search
        - **AI Responses**: Generated using Google Gemini
        - **Voice Output**: Text-to-speech for audio responses
        - **Multi-language**: Supports multiple languages
        
        ### 🛠️ Technology Stack
        - **Frontend**: Streamlit
        - **Speech-to-Text**: OpenAI Whisper
        - **Embeddings**: Sentence Transformers
        - **Vector Search**: FAISS
        - **LLM**: Google Gemini
        - **Text-to-Speech**: Google Text-to-Speech
        
        ### 📝 How to Use
        1. Enter your Gemini API key in the sidebar
        2. Wait for models to load
        3. Upload your audio file in Voice Query tab
        4. Or use Text Query tab for typed questions
        5. Get AI-powered responses with sources
        6. Listen to audio responses
        
        ### 🔗 Get API Key
        [Get your free Gemini API key](https://makersuite.google.com/app/apikey)
        """)


if __name__ == "__main__":
    main()
