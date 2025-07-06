import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter


def readDoc(doc_folder: str = "doc") -> List[str]:
    """
    Loops through all .txt files in the doc folder and returns their text as a list of strings.
    """
    texts = []
    for filename in os.listdir(doc_folder):
        if filename.startswith('.') or filename.startswith('~'):
            continue
        file_path = os.path.join(doc_folder, filename)
        if os.path.isfile(file_path) and filename.lower().endswith('.txt'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    return texts

def chunkDoc() -> List[str]:
    """
    Reads documents and splits them into chunks using RecursiveCharacterTextSplitter.
    """
    docs = readDoc()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    all_splits = []
    for doc in docs:
        splits = text_splitter.split_text(doc)
        all_splits.extend(splits)
    return all_splits

def main():
    chunks = chunkDoc()
    print(f"Total chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks): # Print all chunks for brevity
        print(f"--- Chunk {i+1} ---\n{chunk}\n...")

if __name__ == "__main__":
    main()
