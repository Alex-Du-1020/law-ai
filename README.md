# Law-AI

A Python project for legal document chunking, embedding, and semantic search using ChromaDB and Deepseek LLM.

## Features
- Reads and splits legal documents into manageable text chunks
- Generates embeddings for each chunk using Sentence Transformers
- Stores and queries embeddings with ChromaDB
- Retrieves top relevant contexts for a question and sends them to Deepseek LLM for an answer

## Setup

1. **Clone the repository**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Or manually:
   ```bash
   pip install chromadb sentence-transformers langchain python-dotenv requests
   ```
3. **Prepare your environment:**
   - Place your `.txt` legal documents in the `doc/` folder.
   - Create a `.env` file in the project root with your Deepseek API key:
     ```
     DEEPSEEK_API_KEY=your_actual_deepseek_api_key
     ```

## Usage

### 1. Chunk and Embed Documents
```bash
python src/embedding.py
```
This will read, chunk, embed, and store your documents in ChromaDB.

### 2. Query with a Legal Question
```bash
python src/query.py
```
This will retrieve the top 5 relevant contexts and send them, along with your question, to Deepseek LLM. The answer will be printed.

## File Structure
- `src/chunk.py` – Reads and splits documents
- `src/embedding.py` – Embeds and stores chunks in ChromaDB
- `src/query.py` – Queries ChromaDB and Deepseek LLM
- `doc/` – Place your `.txt` legal documents here
- `.env` – Store your Deepseek API key here (not tracked by git)

## Notes
- Make sure to use Python 3.8+
- The `.gitignore` file ensures sensitive and large files are not tracked

## License
MIT 