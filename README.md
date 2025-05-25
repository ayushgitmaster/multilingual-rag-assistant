# README.md
# 🎤 Voice RAG Assistant

A powerful voice-enabled Retrieval-Augmented Generation (RAG) system that allows users to ask questions using voice input and receive AI-generated responses with source citations.

## ✨ Features

- 🎤 **Voice Input**: Record questions using your microphone
- 🧠 **Smart AI**: Powered by Google Gemini for intelligent responses
- 🔍 **Document Search**: Uses FAISS for fast similarity search
- 🗣️ **Voice Output**: Text-to-speech responses
- 🌍 **Multi-language**: Supports multiple languages
- 📚 **Source Citations**: Shows relevant sources for each response

## 🚀 Quick Start

### Local Development

1. **Clone and install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Get your Gemini API key:**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key

3. **Run the app:**
   ```bash
   streamlit run app1.py
   ```

4. **Open your browser:**
   - Go to `http://localhost:8501`
   - Enter your API key
   - Start asking questions!

### 🌐 Deploy to Streamlit Cloud (FREE)

1. **Fork this repository** on GitHub

2. **Go to [share.streamlit.io](https://share.streamlit.io)**

3. **Click "New app"** and connect your GitHub repository

4. **Set the main file path:** `app.py`

5. **Deploy!** Your app will be live at `https://yourapp.streamlit.app`

## 🛠️ Technology Stack

- **Frontend**: Streamlit for the web interface
- **Speech Recognition**: OpenAI Whisper for voice-to-text
- **Embeddings**: Sentence Transformers for semantic search
- **Vector Database**: FAISS for fast similarity search
- **LLM**: Google Gemini for response generation
- **Text-to-Speech**: Google TTS for voice responses

## 📋 Usage Instructions

1. **Configure API Key**: Enter your Gemini API key in the sidebar
2. **Wait for Loading**: Models will load automatically
3. **Voice Query**: Click the microphone and record your question
4. **Get Response**: AI will provide answers with source citations
5. **Listen**: Audio responses are automatically generated

## 🔧 Customization

### Adding Your Own Documents

Modify the `load_and_process_documents()` function in `app.py`:

```python
documents = [
    {
        'content': "Your document content here...",
        'source': 'Your Source Name'
    },
    # Add more documents...
]
```

### Changing Models

- **Whisper Model**: Change `whisper.load_model("base")` to "small", "medium", or "large"
- **Embedding Model**: Modify the sentence-transformers model name
- **LLM**: Switch to different Gemini models if available

## 📝 API Keys and Security

- **Free Tier**: Gemini API has a generous free tier
- **Security**: Never commit API keys to version control
- **Environment Variables**: Use Streamlit secrets for deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

- **Issues**: Report bugs on GitHub Issues
- **API Help**: Check [Google AI documentation](https://ai.google.dev/)
- **Streamlit Help**: Visit [Streamlit documentation](https://docs.streamlit.io/)

---

**Built with ❤️ using Streamlit and Google AI**
