import os
import json
import hashlib

VECTOR_FILE = 'local_vector_index.json'

def simple_embed(text):
    """Generate a pseudo-semantic pseudo-vector using local hashing 
    (to be swapped with a quantized mobile ONNX/GGML embedding model)."""
    h = hashlib.sha256(text.encode('utf-8')).hexdigest()
    # Convert hex chunks into a deterministic 16-dimension float vector
    vector = [int(h[i:i+2], 16) / 255.0 for i in range(0, 32, 2)]
    return vector

def cosine_similarity(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm1 = sum(a * a for a in v1) ** 0.5
    norm2 = sum(b * b for b in v2) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def save_vector_entry(text, timestamp):
    records = []
    if os.path.exists(VECTOR_FILE):
        try:
            with open(VECTOR_FILE, 'r', encoding='utf-8') as f:
                records = json.load(f)
        except Exception:
            records = []
            
    entry = {
        "time": timestamp,
        "text": text,
        "vector": simple_embed(text)
    }
    records.append(entry)
    
    with open(VECTOR_FILE, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=4)

def query_vector_store(query_text, top_k=3):
    if not os.path.exists(VECTOR_FILE):
        return []
    with open(VECTOR_FILE, 'r', encoding='utf-8') as f:
        records = json.load(f)
        
    q_vec = simple_embed(query_text)
    scored = []
    for r in records:
        score = cosine_similarity(q_vec, r['vector'])
        scored.append((score, r))
        
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:top_k]]
