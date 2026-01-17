#!/usr/bin/env python3
"""
Web Interface for IR System - Gradio-based Query Interface
Integrates with project.ipynb infrastructure: reuses SCORE_CACHE, models, and methods
Supports: BM25, Dense (LaBSE), and Hybrid retrieval methods
"""

import json
import pickle
import os
import ast  # Added for Python dict format parsing
from pathlib import Path
import gradio as gr
import numpy as np
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import jieba
import pandas as pd

# Configuration - same as notebook
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data" / "swim_ir_v1" / "swim_ir_v1"
BASE_DATA_DIR = str(DATA_DIR)

LANGUAGES = ["en", "de", "es", "fr"]  # Removed zh - only has cross-lingual (non-English docs)
K = 10  # Same as notebook
MAX_ITEMS = 10000  # Same as notebook

# Tokenization - same as notebook
def tokenize(text: str, lang_code: str) -> List[str]:
    """Tokenize text based on language (same as notebook)."""
    if lang_code == "zh":
        return list(jieba.cut(text))
    else:
        return text.lower().split()


class IRSystem:
    """
    Unified IR System that reuses the exact implementations from project.ipynb.
    Supports BM25, Dense (LaBSE), and Hybrid retrieval.
    """
    
    def __init__(self):
        self.model = None  # LaBSE model
        self.documents = {}  # lang -> {doc_id: full_text}
        self.doc_ids = {}  # lang -> [doc_id1, doc_id2, ...]
        self.bm25_indices = {}  # lang -> BM25Okapi object
        self.doc_embeddings = {}  # lang -> numpy array
        self.best_alphas = {}  # lang -> best alpha from notebook tuning
        self._initialized = False
        
    def initialize(self):
        """Initialize model and load data (same workflow as notebook)."""
        if self._initialized:
            return
            
        print("🔄 Loading LaBSE model (same as notebook)...")
        self.model = SentenceTransformer("sentence-transformers/LaBSE")
        
        print("🔄 Loading documents and building indices for each language...")
        for lang in LANGUAGES:
            self._load_language_data(lang)
        
        # Load best alpha values from notebook if available
        self._load_best_alphas()
        
        print("✅ Initialization complete!")
        self._initialized = True
    
    def _load_language_data(self, lang: str):
        """Load documents and build indices (same as notebook BM25 cell)."""
        # Use monolingual split - documents are in target language
        # This enables true cross-lingual: English query → target language docs
        # (LaBSE model handles the cross-lingual matching)
        doc_path = DATA_DIR / "monolingual" / lang / "train.jsonl"
        
        if not doc_path.exists():
            print(f"  ⚠️ No data found for {lang}")
            return
        
        # Load documents exactly as notebook does - using load_jsonl_robust logic
        documents = {}
        doc_ids = []
        doc_texts = []
        
        with open(doc_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= MAX_ITEMS:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    # Try JSON first, then Python literal_eval (same as notebook)
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        data = ast.literal_eval(line)
                    
                    doc_id = f"{lang}_{data['_id']}"
                    
                    # Combine title + text (same as notebook)
                    title = data.get("title", "")
                    text = data.get("text", "")
                    full_text = f"{title} {text}".strip()
                    
                    if full_text:
                        documents[doc_id] = full_text
                        doc_ids.append(doc_id)
                        doc_texts.append(full_text)
                    
                except (json.JSONDecodeError, KeyError, ValueError, SyntaxError) as e:
                    continue
        
        if len(doc_texts) == 0:
            print(f"  ⚠️ No valid documents found for {lang}")
            return
        
        self.documents[lang] = documents
        self.doc_ids[lang] = doc_ids
        print(f"  ✓ Loaded {len(doc_texts)} documents for {lang}")
        
        # Build BM25 index (same as notebook)
        print(f"  Building BM25 index for {lang}...")
        tokenized_corpus = [tokenize(doc, lang) for doc in doc_texts]
        self.bm25_indices[lang] = BM25Okapi(tokenized_corpus)
        print(f"  ✓ BM25 index ready for {lang}")
        
        # Load or compute embeddings
        self._load_embeddings(lang, doc_texts)
    
    def _load_embeddings(self, lang: str, doc_texts: List[str]):
        """Load precomputed embeddings from notebook or compute them."""
        # Try loading from notebook's cached embeddings
        embeddings_file = PROJECT_DIR / f"doc_embeddings_{lang}.pkl"
        
        if embeddings_file.exists():
            with open(embeddings_file, "rb") as f:
                self.doc_embeddings[lang] = pickle.load(f)
            print(f"  ✓ Loaded cached embeddings for {lang} ({self.doc_embeddings[lang].shape[0]} docs)")
        else:
            # Compute with same settings as notebook (batch_size=1, normalize=True)
            print(f"  Computing embeddings for {lang} (this may take a while)...")
            embeddings = self.model.encode(
                doc_texts,
                batch_size=1,  # Conservative, same as notebook
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True  # Same as notebook
            )
            self.doc_embeddings[lang] = embeddings
            
            # Cache for future use
            with open(embeddings_file, "wb") as f:
                pickle.dump(embeddings, f)
            print(f"  ✓ Embeddings computed and cached")
    
    def _load_best_alphas(self):
        """Load best alpha values from notebook hybrid tuning if available."""
        # Try to load from CSV file (more compatible than pickle)
        csv_file = PROJECT_DIR / "results" / "rerank" / "df_results_rerank_frozen.csv"
        
        if csv_file.exists():
            try:
                df = pd.read_csv(csv_file)
                hybrid_results = df[df['method'] == 'Hybrid(tuned)']
                
                for _, row in hybrid_results.iterrows():
                    lang = row['language']
                    alpha = row.get('alpha', 0.5)
                    self.best_alphas[lang] = float(alpha)
                
                print(f"  ✓ Loaded tuned alpha values from notebook results")
                for lang, alpha in self.best_alphas.items():
                    print(f"    {lang}: α={alpha:.2f}")
            except Exception as e:
                print(f"  ⚠️ Could not load alpha values: {e}")
                self._set_default_alphas()
        else:
            self._set_default_alphas()
    
    def _set_default_alphas(self):
        """Set default alpha=0.5 for all languages."""
        for lang in LANGUAGES:
            self.best_alphas[lang] = 0.5
        print(f"  Using default α=0.5 for all languages")
    
    def retrieve(self, query: str, language: str, method: str) -> List[Tuple[str, float, str]]:
        """
        Retrieve documents for a query.
        
        Returns:
            List of (doc_id, score, doc_text) tuples, sorted by score (descending)
        """
        if language not in LANGUAGES:
            return [("error", 0, "Error: Language not supported")]
        
        if not self._initialized:
            return [("error", 0, "Error: System not initialized")]
        
        if language not in self.documents or len(self.documents[language]) == 0:
            return [("error", 0, f"Error: No documents loaded for language '{language}'. Check data path.")]
        
        # Route to appropriate method
        if method == "Dense only (LaBSE)":
            return self._retrieve_dense(query, language)
        elif method == "BM25 only":
            return self._retrieve_bm25(query, language)
        elif method.startswith("Hybrid"):
            return self._retrieve_hybrid(query, language)
        else:
            return [("error", 0, "Error: Unknown retrieval method")]
    
    def _retrieve_bm25(self, query: str, language: str) -> List[Tuple[str, float, str]]:
        """BM25 retrieval (same as notebook lines 436-446)."""
        if language not in self.bm25_indices:
            return [("error", 0, "Error: BM25 index not available")]
        
        # Tokenize query (same as notebook)
        tokenized_query = tokenize(query, language)
        
        # Get BM25 scores (same as notebook)
        bm25 = self.bm25_indices[language]
        scores = np.asarray(bm25.get_scores(tokenized_query), dtype=np.float32)
        
        # Get top-K (same as notebook)
        top_indices = scores.argsort()[-K:][::-1]
        doc_ids_list = self.doc_ids[language]
        
        results = [
            (doc_ids_list[idx], float(scores[idx]), self.documents[language][doc_ids_list[idx]])
            for idx in top_indices
        ]
        return results
    
    def _retrieve_dense(self, query: str, language: str) -> List[Tuple[str, float, str]]:
        """Dense retrieval using LaBSE (same as notebook lines 709-730)."""
        if language not in self.doc_embeddings:
            return [("error", 0, "Error: Embeddings not available")]
        
        # Encode query (same as notebook)
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True  # Same as notebook
        )
        
        doc_embeddings = self.doc_embeddings[language]
        
        # Compute cosine similarities (same as notebook line 723)
        query_emb_2d = query_embedding.reshape(1, -1)
        similarities = cosine_similarity(query_emb_2d, doc_embeddings)[0]
        
        # Get top-K
        top_indices = np.argsort(similarities)[::-1][:K]
        doc_ids_list = self.doc_ids[language]
        
        results = [
            (doc_ids_list[idx], float(similarities[idx]), self.documents[language][doc_ids_list[idx]])
            for idx in top_indices
        ]
        return results
    
    def _retrieve_hybrid(self, query: str, language: str) -> List[Tuple[str, float, str]]:
        """
        Hybrid retrieval combining BM25 and Dense (same as notebook lines 1015-1063).
        Uses min-max normalization and learned alpha.
        """
        if language not in self.bm25_indices or language not in self.doc_embeddings:
            return [("error", 0, "Error: Hybrid retrieval not available")]
        
        # Get BM25 scores
        tokenized_query = tokenize(query, language)
        bm25 = self.bm25_indices[language]
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # Get Dense scores
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        doc_embeddings = self.doc_embeddings[language]
        query_emb_2d = query_embedding.reshape(1, -1)
        dense_scores = cosine_similarity(query_emb_2d, doc_embeddings)[0]
        
        # Ensure both score arrays have the same length
        # This handles cases where embeddings were cached with different MAX_ITEMS
        min_len = min(len(bm25_scores), len(dense_scores))
        bm25_scores = bm25_scores[:min_len]
        dense_scores = dense_scores[:min_len]
        
        # Normalize scores (same as notebook lines 1050-1058)
        bm25_min, bm25_max = bm25_scores.min(), bm25_scores.max()
        dense_min, dense_max = dense_scores.min(), dense_scores.max()
        
        if bm25_max > bm25_min:
            bm25_norm = (bm25_scores - bm25_min) / (bm25_max - bm25_min)
        else:
            bm25_norm = bm25_scores
            
        if dense_max > dense_min:
            dense_norm = (dense_scores - dense_min) / (dense_max - dense_min)
        else:
            dense_norm = dense_scores
        
        # Combine with alpha (same as notebook lines 1060-1061)
        alpha = self.best_alphas.get(language, 0.5)
        hybrid_scores = alpha * dense_norm + (1 - alpha) * bm25_norm
        
        # Get top-K
        top_indices = np.argsort(hybrid_scores)[::-1][:K]
        doc_ids_list = self.doc_ids[language]
        
        results = [
            (doc_ids_list[idx], float(hybrid_scores[idx]), self.documents[language][doc_ids_list[idx]])
            for idx in top_indices
        ]
        return results


# Initialize system
ir_system = IRSystem()


def query_interface(query: str, language: str, method: str) -> str:
    """Gradio interface function."""
    if not query.strip():
        return "⚠️ Please enter a query"
    
    if not ir_system._initialized:
        ir_system.initialize()
    
    try:
        results = ir_system.retrieve(query, language, method)
        
        # Format output
        alpha_info = ""
        if method.startswith("Hybrid") and language in ir_system.best_alphas:
            alpha_info = f" (α={ir_system.best_alphas[language]:.2f})"
        
        output = f"**Query:** {query}\n"
        output += f"**Method:** {method}{alpha_info} | **Language:** {language.upper()}\n"
        output += f"**Retrieved {len(results)} documents**\n"
        output += "---\n\n"
        
        if results and results[0][0] == "error":
            return output + results[0][2]
        
        for i, (doc_id, score, doc_text) in enumerate(results, 1):
            # Truncate long documents
            doc_preview = doc_text[:400] + "..." if len(doc_text) > 400 else doc_text
            output += f"### {i}. Score: {score:.4f}\n"
            output += f"*Doc ID:* `{doc_id}`\n\n"
            output += f"{doc_preview}\n\n"
            output += "---\n\n"
        
        return output
    
    except Exception as e:
        import traceback
        return f"❌ Error: {str(e)}\n\n```\n{traceback.format_exc()}\n```"


def main():
    """Launch Gradio interface."""
    print("="*70)
    print("🚀 IR SYSTEM WEB INTERFACE")
    print("="*70)
    print("📚 Reusing infrastructure from project.ipynb")
    print(f"📊 Same BM25, LaBSE, and Hybrid implementations")
    print(f"🎯 Evaluated on {MAX_ITEMS} docs/language across {len(LANGUAGES)} languages")
    print("="*70 + "\n")
    
    # Fixed for Gradio 6.0 - removed theme from Blocks constructor
    with gr.Blocks(title="IR System - Query Interface") as demo:
        gr.Markdown("# 🔍 Multilingual Information Retrieval System")
        gr.Markdown("""
        Query documents using **BM25** (sparse), **LaBSE** (dense), or **Hybrid** retrieval.  
        *Powered by the same models and methods evaluated in project.ipynb*
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                query_input = gr.Textbox(
                    label="Query",
                    placeholder="Enter your search query in any supported language...",
                    lines=3
                )
            with gr.Column(scale=1):
                language_dropdown = gr.Dropdown(
                    choices=LANGUAGES,
                    value="en",
                    label="Document Language",
                    info="Language of documents to search"
                )
                method_dropdown = gr.Dropdown(
                    choices=["Dense only (LaBSE)", "BM25 only", "Hybrid (α tuned)"],
                    value="Dense only (LaBSE)",
                    label="Retrieval Method"
                )
        
        search_button = gr.Button("🔍 Search", variant="primary", size="lg")
        
        results_output = gr.Markdown(
            value="*Results will appear here...*",
            label="Results"
        )
        
        # Examples from different languages
        gr.Examples(
            examples=[
                ["What are the health benefits of exercise?", "en", "Dense only (LaBSE)"],
                ["Wie funktioniert künstliche Intelligenz?", "de", "Hybrid (α tuned)"],
                ["¿Cuáles son los efectos del cambio climático?", "es", "BM25 only"],
                ["Comment préparer une recette traditionnelle?", "fr", "Dense only (LaBSE)"],
                ["什么是机器学习?", "en", "Hybrid (α tuned)"],
            ],
            inputs=[query_input, language_dropdown, method_dropdown],
            label="📝 Example Queries"
        )
        
        search_button.click(
            fn=query_interface,
            inputs=[query_input, language_dropdown, method_dropdown],
            outputs=results_output
        )
        
        # Allow Enter key to trigger search
        query_input.submit(
            fn=query_interface,
            inputs=[query_input, language_dropdown, method_dropdown],
            outputs=results_output
        )
    
    # Fixed: moved theme to launch() for Gradio 6.0, and use server_port=None to auto-find available port
    demo.launch(
        share=False, 
        server_name="127.0.0.1", 
        server_port=None,  # Auto-find available port
        theme=gr.themes.Soft()
    )


if __name__ == "__main__":
    main()
