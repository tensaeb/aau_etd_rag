
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from pathlib import Path
import requests
import time
import yaml
import pickle
from rank_bm25 import BM25Okapi

class RAGPipeline:
    def __init__(self, config_path="config.yaml"):
        """
        Initializes the RAG pipeline by loading all necessary models and data.
        """
        print("Initializing RAG pipeline...")
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Load all resources
        self._load_resources()
        print("RAG pipeline initialized successfully.")

    def _load_resources(self):
        """Loads models, FAISS index, BM25 index, and metadata from disk."""
        # Paths
        self.chunk_dir = Path(self.config["data"]["chunk_dir"])
        faiss_index_path = self.config["faiss"]["index_path"]
        bm25_index_path = Path(faiss_index_path).parent / "bm25_index.pkl"
        metadata_path = Path(self.config["data"]["embedding_dir"]) / self.config["faiss"]["metadata_file"]

        # Models
        print("  - Loading embedding model...")
        self.embed_model = SentenceTransformer(self.config["models"]["embedding"])
        print("  - Loading cross-encoder model...")
        self.cross_encoder = CrossEncoder(self.config["models"]["cross_encoder"])

        # Data and Indexes
        print("  - Loading FAISS index and metadata...")
        self.index = faiss.read_index(faiss_index_path)
        self.metadata = np.load(metadata_path, allow_pickle=True)

        print("  - Loading BM25 index...")
        with open(bm25_index_path, "rb") as f:
            bm25_data = pickle.load(f)
            self.bm25 = bm25_data["bm25"]
            self.bm25_doc_mapping = bm25_data["doc_mapping"]

    def _load_chunk(self, source, chunk_id):
        """Loads a specific text chunk from a file."""
        file_path = self.chunk_dir / f"{source}.txt"

        if not file_path.exists():
            return f"[Error: Chunk file not found: {file_path}]"

        try:
            chunks = file_path.read_text(encoding="utf-8").split("\n\n---\n\n")
            if chunk_id >= len(chunks):
                return f"[Error: Chunk ID {chunk_id} out of range for file {file_path}]"
            return chunks[chunk_id]
        except Exception as e:
            return f"[Error loading chunk: {e}]"

    def _retrieve(self, query):
        """
        Retrieves initial document candidates using a hybrid approach:
        - FAISS for semantic search.
        - BM25 for keyword search.
        - Results are combined with Reciprocal Rank Fusion (RRF).
        """
        k = self.config["rag"]["retrieval_candidates"]

        # 1. Semantic Search (FAISS)
        q_vec = self.embed_model.encode([f"query: {query}"])
        _, faiss_indices = self.index.search(q_vec, k)
        faiss_results = [self.metadata[i] for i in faiss_indices[0]]

        # 2. Keyword Search (BM25)
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_n_indices = np.argsort(bm25_scores)[::-1][:k]
        bm25_results = [self.bm25_doc_mapping[i] for i in top_n_indices]

        # 3. Reciprocal Rank Fusion (RRF)
        return self._reciprocal_rank_fusion([faiss_results, bm25_results])

    def _reciprocal_rank_fusion(self, result_lists, k=60):
        """Combines multiple ranked lists using RRF."""
        ranked_items = {}
        for results in result_lists:
            for rank, result in enumerate(results):
                # Create a unique key for each document
                doc_key = (result['source'], result['chunk_id'])
                if doc_key not in ranked_items:
                    ranked_items[doc_key] = 0
                ranked_items[doc_key] += 1 / (k + rank + 1)

        # Sort items by their RRF score in descending order
        sorted_items = sorted(ranked_items.items(), key=lambda item: item[1], reverse=True)

        # Convert back to the original metadata format
        final_results = [{"source": key[0], "chunk_id": key[1]} for key, score in sorted_items]
        return final_results

    def _rerank(self, query, items):
        """Reranks retrieved items using the cross-encoder."""
        if not items:
            return []

        pairs = [(query, self._load_chunk(item["source"], item["chunk_id"])) for item in items]
        scores = self.cross_encoder.predict(pairs)

        scored_items = sorted(zip(items, scores), key=lambda x: x[1], reverse=True)
        return [item for item, score in scored_items]

    def _truncate_text(self, text, max_chars):
        """Truncates text intelligently at sentence boundaries."""
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars]
        last_break = max(truncated.rfind('.'), truncated.rfind('\n'))
        if last_break > max_chars * 0.8:
            return truncated[:last_break + 1] + "..."
        return truncated + "..."

    def _build_context(self, items):
        """Builds the final context string for the LLM prompt."""
        max_chars = self.config["rag"]["max_context_chars"]
        max_chunk_chars = max_chars // len(items) if items else max_chars

        blocks = []
        total_chars = 0

        for item in items:
            text = self._load_chunk(item["source"], item["chunk_id"])
            if len(text) > max_chunk_chars:
                text = self._truncate_text(text, max_chunk_chars)

            block = f"[{item['source']} | chunk {item['chunk_id']}]\n{text}"

            if total_chars + len(block) > max_chars and blocks:
                break

            blocks.append(block)
            total_chars += len(block) + 2

        return "\n\n".join(blocks)

    def _print_gpu_error(self):
        """Prints GPU error instructions."""
        print("\n" + "="*70 + "\nGPU DEVICE ERROR DETECTED\n" + "="*70)
        print("Your GPU may be losing connection or running out of memory.")
        print("\nSOLUTIONS (try in order):")
        print("  1. In LM Studio: Set 'GPU Offload' to 0 (CPU mode)")
        print("  2. Restart LM Studio completely")
        print("  3. Update your GPU drivers\n" + "="*70 + "\n")

    def _print_cache_error(self):
        """Prints KV cache error instructions."""
        print("\n" + "="*70 + "\nKV CACHE ERROR DETECTED\n" + "="*70)
        print("This is an LM Studio internal issue. Retrying won't help.")
        print("\nSOLUTION: You must manually fix this in LM Studio:")
        print("  1. Restart LM Studio completely, OR")
        print("  2. Stop the model, then start it again\n" + "="*70 + "\n")

    def _check_for_errors(self, error_text):
        """Checks if error text indicates GPU or cache issues."""
        error_lower = error_text.lower()
        if "devicelost" in error_lower or "vk::device" in error_lower or "fencestatus" in error_lower:
            return "gpu"
        if ("cache" in error_lower or "sequence" in error_lower or "position" in error_lower or "inconsistent" in error_lower):
            return "cache"
        if "channel error" in error_lower:
            return "channel"
        return None

    def _ask_lmstudio(self, prompt, max_retries=3):
        """Sends a prompt to the LM Studio API with robust error handling."""
        lm_config = self.config["lm_studio"]
        payload = {
            "model": lm_config["model"],
            "messages": [{"role": "user", "content": f"You are a factual academic assistant.\n\n{prompt}"}],
            "temperature": lm_config["temperature"],
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(lm_config["url"], json=payload, timeout=120)
                result = response.json()

                if "error" in result:
                    error_type = self._check_for_errors(str(result.get("error", "")))
                    if error_type == "gpu":
                        self._print_gpu_error()
                        return "GPU device error. See console for details."
                    elif error_type == "cache":
                        self._print_cache_error()
                        return "KV cache error. See console for details."

                response.raise_for_status()

                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]

                if attempt < max_retries - 1:
                    print(f"Unexpected API response, retrying... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(3 * (attempt + 1))
                else:
                    return f"Unexpected API response: {result}"

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(f"Request timed out, retrying... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(3 * (attempt + 1))
                else:
                    return "Error: Request timed out. Try restarting LM Studio."

            except requests.exceptions.RequestException as e:
                error_text = str(e)
                try:
                    error_text += str(e.response.json())
                except:
                    pass

                error_type = self._check_for_errors(error_text)
                if error_type == "gpu":
                    self._print_gpu_error()
                    return "GPU device error. See console for details."
                elif error_type == "cache":
                    self._print_cache_error()
                    return "KV cache error. See console for details."
                elif error_type == "channel" and hasattr(e.response, 'status_code') and e.response.status_code >= 500:
                    self._print_gpu_error()
                    return "Channel error (likely GPU issue). See console for details."

                if attempt < max_retries - 1:
                    print(f"Request error, retrying... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(3 * (attempt + 1))
                else:
                    return f"Error calling LM Studio API: {e}"

        return "Error: All retry attempts failed."

    def ask(self, query):
        """
        Main method to ask a question to the RAG pipeline.
        Orchestrates retrieval, reranking, context building, and answer generation.
        """
        print(f"\nProcessing question: {query}")

        # 1. Retrieve
        print("  - Retrieving initial candidates...")
        retrieved_items = self._retrieve(query)

        # 2. Rerank
        print("  - Reranking retrieved documents...")
        reranked_items = self._rerank(query, retrieved_items)
        top_k = self.config["rag"]["top_k"]
        final_items = reranked_items[:top_k]

        # 3. Build Context
        print("  - Building context...")
        context = self._build_context(final_items)

        # 4. Build Prompt
        prompt = f"""Answer ONLY using the context below.
If the answer is not present, say so.

Context:
{context}

Question:
{query}

Answer:
"""

        # 5. Ask LLM
        print("  - Sending prompt to LM Studio...")
        answer = self._ask_lmstudio(prompt)

        return answer
