"""
RAG Pipeline for LM Studio

Retrieves relevant passages from ETD documents and generates answers using LM Studio.
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import requests
import time

# Configuration
LMSTUDIO_URL = "http://192.168.1.22:1234/v1/chat/completions"
INDEX_PATH = "indexes/aau_faiss.index"
METADATA_PATH = "data/embeddings/metadata.npy"
CHUNK_DIR = Path("data/chunks")
EMBED_MODEL_NAME = "intfloat/e5-base-v2"

# Load resources
index = faiss.read_index(INDEX_PATH)
metadata = np.load(METADATA_PATH, allow_pickle=True)
embed_model = SentenceTransformer(EMBED_MODEL_NAME)


def load_chunk(source, chunk_id):
    """Load a specific chunk from a chunk file."""
    file = CHUNK_DIR / f"{source}.txt"
    if not file.exists():
        return f"[Error: Chunk file not found: {file}]"
    chunks = file.read_text(encoding="utf-8").split("\n\n---\n\n")
    if chunk_id >= len(chunks):
        return f"[Error: Chunk ID {chunk_id} out of range]"
    return chunks[chunk_id]


def retrieve(query, k=3):
    """Retrieve k most relevant chunks for a query."""
    q_vec = embed_model.encode([f"query: {query}"])
    _, I = index.search(q_vec, k)
    return [metadata[i] for i in I[0]]


def truncate_text(text, max_chars):
    """Truncate text to max_chars, trying to break at sentence boundaries."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_period = truncated.rfind('.')
    last_newline = truncated.rfind('\n')
    break_point = max(last_period, last_newline)
    if break_point > max_chars * 0.8:
        return truncated[:break_point + 1] + "..."
    return truncated + "..."


def build_context(items, max_context_chars=6000):
    """Build context from retrieved items, truncating if necessary."""
    blocks = []
    total_chars = 0
    max_chunk_chars = max_context_chars // len(items) if items else max_context_chars
    
    for item in items:
        text = load_chunk(item["source"], item["chunk_id"])
        if len(text) > max_chunk_chars:
            text = truncate_text(text, max_chunk_chars)
        
        block = f"[{item['source']} | chunk {item['chunk_id']}]\n{text}"
        
        if total_chars + len(block) > max_context_chars and blocks:
            break
        
        blocks.append(block)
        total_chars += len(block) + 2
    
    return "\n\n".join(blocks)


def _print_gpu_error():
    """Print GPU error instructions."""
    print("\n" + "="*70)
    print("GPU DEVICE ERROR DETECTED")
    print("="*70)
    print("Your GPU is losing connection or running out of memory.")
    print("\nSOLUTIONS (try in order):")
    print("  1. In LM Studio: Set 'GPU Offload' to 0 (CPU mode)")
    print("  2. Restart LM Studio completely")
    print("  3. Update your GPU drivers")
    print("="*70 + "\n")


def _print_cache_error():
    """Print KV cache error instructions."""
    print("\n" + "="*70)
    print("KV CACHE ERROR DETECTED")
    print("="*70)
    print("This is an LM Studio internal issue. Retrying won't help.")
    print("\nSOLUTION: You must manually fix this in LM Studio:")
    print("  1. Restart LM Studio completely, OR")
    print("  2. Stop the model, then start it again")
    print("="*70 + "\n")


def _check_for_errors(error_text):
    """Check if error text indicates GPU or cache issues."""
    error_lower = error_text.lower()
    if "devicelost" in error_lower or "vk::device" in error_lower or "fencestatus" in error_lower:
        return "gpu"
    if ("cache" in error_lower or "sequence" in error_lower or 
        "position" in error_lower or "inconsistent" in error_lower):
        return "cache"
    if "channel error" in error_lower:
        return "channel"
    return None


def ask_lmstudio(prompt, max_retries=3):
    """Ask LM Studio with retry logic and error handling."""
    time.sleep(0.5)  # Small delay to help clear stale cache
    
    full_prompt = "You are a factual academic assistant.\n\n" + prompt
    payload = {
        "model": "mistral",
        "messages": [{"role": "user", "content": full_prompt}],
        "temperature": 0.0
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(LMSTUDIO_URL, json=payload, timeout=120)
            
            # Check response body for errors
            try:
                result = response.json()
            except:
                result = {}
            
            # Check for errors in response body
            if "error" in result:
                error_type = _check_for_errors(str(result.get("error", "")))
                if error_type == "gpu":
                    _print_gpu_error()
                    return "GPU device error. Please switch LM Studio to CPU mode (set GPU Offload to 0)."
                elif error_type == "cache":
                    _print_cache_error()
                    return "KV cache error. Please restart LM Studio or stop/restart the model."
            
            response.raise_for_status()
            
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            
            if attempt < max_retries - 1:
                print(f"Unexpected API response, retrying... (attempt {attempt + 1}/{max_retries})")
                time.sleep(3 * (attempt + 1))
                continue
            return f"Unexpected API response: {result}"
            
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                print(f"Request timed out, retrying... (attempt {attempt + 1}/{max_retries})")
                time.sleep(3 * (attempt + 1))
                continue
            return "Error: Request timed out. Try restarting LM Studio."
            
        except requests.exceptions.HTTPError as e:
            error_text = ""
            try:
                error_text = str(e.response.json())
            except:
                error_text = e.response.text[:500] if hasattr(e.response, 'text') else str(e)
            
            error_type = _check_for_errors(str(e) + error_text)
            if error_type == "gpu":
                _print_gpu_error()
                return "GPU device error. Please switch LM Studio to CPU mode (set GPU Offload to 0)."
            elif error_type == "cache":
                _print_cache_error()
                return "KV cache error. Please restart LM Studio or stop/restart the model."
            elif error_type == "channel" and e.response.status_code >= 500:
                _print_gpu_error()
                return "Channel error (likely GPU issue). Please switch LM Studio to CPU mode."
            
            return f"HTTP Error {e.response.status_code}: {e}"
            
        except requests.exceptions.RequestException as e:
            error_type = _check_for_errors(str(e))
            if error_type == "gpu":
                _print_gpu_error()
                return "GPU device error. Please switch LM Studio to CPU mode (set GPU Offload to 0)."
            elif error_type == "cache":
                _print_cache_error()
                return "KV cache error. Please restart LM Studio or stop/restart the model."
            
            if attempt < max_retries - 1:
                print(f"Request error, retrying... (attempt {attempt + 1}/{max_retries})")
                time.sleep(3 * (attempt + 1))
                continue
            return f"Error calling LM Studio API: {e}"
    
    return "Error: All retry attempts failed. Please check LM Studio and try again."


if __name__ == "__main__":
    # ============================================================================
    # UPDATE YOUR QUESTION HERE:
    # ============================================================================
    # Based on your documents (SDN, DDoS detection, network management, security):
    # 
    # Example questions you can ask:
    # - "What is Software-Defined Networking (SDN) and how does it work?"
    # - "How can DDoS attacks be detected in Software-Defined Networks?"
    # - "What are the advantages of distributed SDN controller architecture?"
    # - "What security threats exist in SDN controllers?"
    # - "How does flow-based feature analysis help in DDoS detection?"
    # - "What is the STRIDE threat model and how is it used for SDN security?"
    # - "What are the challenges in managing large-scale networks?"
    # - "How do attack trees help in security analysis of SDN controllers?"
    #
    # ============================================================================
    question = "What is Software-Defined Networking (SDN) and how does it work?"
    
    # You can also adjust the number of retrieved chunks (k=2 means top 2 results)
    retrieved = retrieve(question, k=2)
    context = build_context(retrieved)
    
    prompt = f"""Answer ONLY using the context below.
If the answer is not present, say so.

Context:
{context}

Question:
{question}

Answer:
"""
    
    answer = ask_lmstudio(prompt)
    print("\nANSWER:\n", answer)
