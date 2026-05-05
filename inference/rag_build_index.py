import os
import json
import faiss
from sentence_transformers import SentenceTransformer

# paths
CORPUS_PATH = "rag/rag_corpus/rag.jsonl"
INDEX_DIR = "rag_index"
os.makedirs(INDEX_DIR, exist_ok=True)

INDEX_PATH = os.path.join(INDEX_DIR, "rag_index.faiss")
META_PATH = os.path.join(INDEX_DIR, "rag_index_meta.json")

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # embedding model


def load_corpus(path):
    '''
    Load corpus from path
    '''
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


def main():
    print(f"Loading corpus from {CORPUS_PATH} ...")
    docs = load_corpus(CORPUS_PATH)
    print(f"Loaded {len(docs)} docs")

    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    texts = [d["text"] for d in docs]
    print("Encoding documents...")
    embeddings = embedder.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))

    print(f"Saving FAISS index to {INDEX_PATH}")
    faiss.write_index(index, INDEX_PATH)

    meta = {
        "docs": docs,
        "embedding_model": EMBED_MODEL_NAME,
    }
    print(f"Saving metadata to {META_PATH}")
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Done building index.")


if __name__ == "__main__":
    main()
