import dis
import google.genai as genai
import os
from dotenv import load_dotenv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import argparse

import numpy as np

from utils import CorpusRow, load_corpus, cosine_similarity, embed_query_bge_m3

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()

load_dotenv()



# ------------



def rank_against_corpus(
    q_vec: np.ndarray,
    corpus: List[CorpusRow],
    *,
    top_k: int = 10,
    source_id: str | None = None,
) -> Dict[str, List[Tuple[CorpusRow, float]]]:

    if source_id is not None: 
        filtered_corpus = [r for r in corpus if r.metadata.get("source_id") == source_id]
    else:
        filtered_corpus = corpus

    cos_scores = np.array([cosine_similarity(q_vec, r.vector) for r in filtered_corpus], dtype=np.float32)

    # Get top_k indices
    k = min(top_k, len(filtered_corpus))

    cos_idx = np.argpartition(-cos_scores, kth=k-1)[:k]
    cos_idx = cos_idx[np.argsort(-cos_scores[cos_idx])]

    cosine_rank = [(filtered_corpus[i], float(cos_scores[i])) for i in cos_idx]

    return cosine_rank

# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", type=str, required=True)
    ap.add_argument("--source_id", type=str, default=None)

    args = ap.parse_args()
    
    ask_gemini(args.query, args.source_id)

def ask_gemini(query: str, source_id: str | None = None) -> None:

    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    q_vec = embed_query_bge_m3(
            query,
            model_name="BAAI/bge-m3",
            use_fp16=True,
            max_length=256,
        )
    
    #print("Query vector:", q_vec)
    corpus = load_corpus("data/corpus_tr_bge_m3.jsonl")

    rankings = rank_against_corpus(q_vec, corpus, top_k=3, source_id=source_id)
    
    """
    print("Top rankings:")
    for row, score in rankings:
        print(f"ID: {row.id}, Score: {score:.4f}, Text: {row.text[:100]}..., Metadata: {row.metadata}")
    """

    chunks = [r.text for r, s in rankings if s > 0.5]

    context = "Context:\n" + "\n\n".join([f"- {chunk}" for chunk in chunks])
    final_prompt = f"""
        Aşağıdaki "KAYNAKLAR" bölümünde verilen bilgileri kullanarak, "SORU"yu yanıtla. Eğer kaynaklar soruyu yanıtlamak için yeterli değilse, "Bilmiyorum" diye cevap ver.
        KAYNAKLAR:
        {context}

        SORU:
        {query}

        Cevap:
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=final_prompt
    )

    print("----------- Gemini Yanıtı -----------")
    print(response.text)

if __name__ == "__main__":
    main()