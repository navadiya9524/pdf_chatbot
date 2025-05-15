# PDF Chatbot

A conversational AI application that allows users to chat with their PDF documents. This application uses LangChain, Google Gemini embeddings, and a FAISS vector database to enable semantic search and intelligent responses based on document content.

## Features

* PDF document uploading and processing
* Text extraction and chunking for efficient processing
* Vector embeddings for semantic search capabilities
* Conversational memory to maintain context in chats
* Support for multiple PDF uploads
* Document-grounded responses to user queries

## Tech Stack

* **Backend:** Python, LangChain
* **Language Model:** Google Gemini 2.0 Flash
* **Embeddings:** Text Embedding 004 model
* **Vector Database:** FAISS

## Installation

### Prerequisites

* Python 3.10 or higher
* Google API key for Gemini access

### Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/navadiya9524/pdf_chatbot.git
   cd pdf_chatbot
   ```

2. **Create a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\\Scripts\\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your Google API key**:

   * Create a `.env` file in the project root
   * Add your API key:

     ```text
     GOOGLE_API_KEY=your_api_key_here
     ```

## Usage

1. **Run the application**:

   ```bash
   python app.py
   ```

2. **Follow the command-line prompts** to interact with your PDF documents:

   * Upload PDF documents as directed by the application
   * Ask questions about your documents through the interface

## Project Structure

```
pdf_chatbot/
├── app.py                 # Main application
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## How It Works

### Document Processing:

1. PDFs are uploaded and processed
2. Text is extracted and divided into manageable chunks
3. Chunks are embedded using Gemini's Text Embedding 004 model

### Vector Storage:

* Text embeddings are stored in a FAISS vector database
* This enables semantic search capabilities

### User Queries:

1. When a user asks a question, it's converted to an embedding
2. Similar document chunks are retrieved from the vector store
3. Retrieved context is used to generate a response using Google Gemini 2.0 Flash model

### Conversation Memory:

* The application maintains context across multiple interactions
* This enables follow-up questions and more natural conversations

## Acknowledgments

* LangChain for the document processing and LLM interaction framework
* Google Gemini for the language model
* Text Embedding Models for the embedding capabilities
