from dotenv import load_dotenv
import os
import chromadb
from sentence_transformers import SentenceTransformer
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Initialize ChromaDB persistent client and collection
chroma_client = chromadb.PersistentClient(path="chromadb_data")
collection = chroma_client.get_or_create_collection("law_texts")

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def query_db(question: str):
    """
    Embeds the question, queries ChromaDB, and returns the top 5 related contexts (documents).
    """
    query_vector = model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=5
    )
    # Extract the top 5 related documents (contexts)
    contexts = results.get('documents', [[]])[0]
    return contexts

def ask_deepseek(question: str, contexts: list) -> str:
    """
    Sends the question and contexts to Deepseek LLM API and returns the answer.
    """
    DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"  # Example endpoint
    if not API_KEY:
        return "Error: DEEPSEEK_API_KEY not set in environment."

    # Format the prompt
    context_str = "\n\n".join(contexts)
    prompt = f"参考以下内容回答问题：\n{context_str}\n\n问题：{question}"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",  # Replace with your model name if needed
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        # Adjust this depending on Deepseek's response format
        return result['choices'][0]['message']['content']
    else:
        return f"Error: {response.status_code} {response.text}"

app = FastAPI()

@app.post("/query")
async def query_endpoint(request: Request):
    data = await request.json()
    question = data.get("question")
    if not question:
        return JSONResponse(status_code=400, content={"error": "Missing 'question' in request body."})
    contexts = query_db(question)
    answer = ask_deepseek(question, contexts)
    print("\nDeepseek Answer:")
    return {"question": question, "contexts": contexts, "answer": answer}

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    # question = input("请输入您的法律问题: ")
    question = "河南省南阳市人民政府需要赔偿多少钱"
    contexts = query_db(question)
    print("Top 5 related contexts:")
    for i, ctx in enumerate(contexts, 1):
        print(f"{i}. {ctx[:200]}...")
    print("\nSending to Deepseek...")
    answer = ask_deepseek(question, contexts)
    print("\nDeepseek Answer:")
    print(answer)
    # To run the API: uvicorn src.query:app --reload
