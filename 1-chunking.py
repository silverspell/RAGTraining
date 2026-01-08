from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Iterable
import hashlib
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from FlagEmbedding import BGEM3FlagModel

@dataclass
class Chunk:
    id: str
    text: str
    metadata: Dict[str, Any]


def _stable_chunk_id(source_id: str, chunk_index: int, text: str) -> str:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{source_id}:{chunk_index}:{h}"

def build_text_splitter(
        *,
        encoding_name: str = "o200k_base",
        chunk_size: int = 600,   # token cinsinden hedef
        chunk_overlap: int = 80, # token cinsinden overlap
) -> tuple[RecursiveCharacterTextSplitter, Any]:
    enc = tiktoken.get_encoding(encoding_name)

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=encoding_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",         # paragraf
            "\n",           # satır
            ". ",           # cümle
            "? ",
            "! ",
            "; ",
            ": ",
            ", ",
            " ",
            ""              # last resort: karakter bazlı
        ],
        add_start_index=True,
    )

    return splitter, enc


def chunk_documents(
        docs: Iterable[Dict[str, Any]],
        *,
        encoding_name: str = "o200k_base",
        chunk_size: int = 600,
        chunk_overlap: int = 80,
        text_key: str = "text",
        source_id_key: str = "source_id",
) -> List[Chunk]:
    """
    docs input örneği:
      {"source_id": "policy_001", "text": "...", "title": "...", "url": "..."}
    """
    splitter, enc = build_text_splitter(
        encoding_name=encoding_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    out: List[Chunk] = []

    for doc in docs:
        source_id = str(doc.get(source_id_key) or doc.get("id") or "unknown")
        text = str(doc.get(text_key) or "")

        pieces = splitter.create_documents([text], metadatas=[{k: v for k, v in doc.items() if k != text_key}])
        for i, p in enumerate(pieces):
            chunk_text = p.page_content.strip()
            if not chunk_text:
                continue

            token_count = len(enc.encode(chunk_text))
            md = dict(p.metadata or {})
            md.update({
                "source_id": source_id,
                "chunk_index": i,
                "token_count": token_count,
            })

            chunk_id = _stable_chunk_id(source_id, i, chunk_text)
            out.append(Chunk(id=chunk_id, text=chunk_text, metadata=md))
    return out

def print_chunk_stats(chunks: List[Chunk], top_n: int = 5) -> None:
    if not chunks:
        print("No chunks.")
        return

    token_counts = [c.metadata["token_count"] for c in chunks]
    print(f"chunks: {len(chunks)}")
    print(f"min/avg/max tokens: {min(token_counts)} / {sum(token_counts)/len(token_counts):.1f} / {max(token_counts)}")

    # En uzun birkaç chunk'a bak (önizleme)
    longest = sorted(chunks, key=lambda c: c.metadata["token_count"], reverse=True)[:top_n]
    print("\nLongest chunks:")
    for c in longest:
        preview = c.text[:160].replace("\n", " ")
        print(f"- {c.id} | {c.metadata['token_count']} tok | {preview}...")

# Örnek kullanım
if __name__ == "__main__":
    sample_docs = [
        {
            "source_id": "policy_returns_tr",
            "title": "İade Politikası",
            "text": (
                "İade politikamız aşağıdaki gibidir.\n\n"
                "1) Teslimat tarihinden itibaren 14 gün içinde iade talebi oluşturabilirsiniz. "
                "Ürün kullanılmamış olmalıdır.\n\n"
                "2) Kargo takip numarası sipariş sayfasında yer alır. "
                "İade kargo bedeli kampanya dönemlerinde ücretsiz olabilir.\n\n"
                "3) İade onayı sonrasında ücret iadesi 3-10 iş günü içinde gerçekleştirilir.\n"
            )
        }
    ]

    chunks = chunk_documents(
        sample_docs,
        encoding_name="o200k_base",
        chunk_size=30,       # demo için küçük
        chunk_overlap=3,
    )
    
    print_chunk_stats(chunks)
    print("----------------------------------------------")
    """
    print("\n--- Example chunk ---")
    print(chunks[0].metadata)
    print(chunks[0].text)
    """


    
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True) # GPU varsa fp16 kullan
    docs = [c.text for c in chunks]
    doc_out = model.encode(
        docs,
        batch_size=32,
        max_length=1024,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False
    )

    doc_vecs = doc_out["dense_vecs"] # (N, 1024)
    for i in range(len(docs)):
        print(f"Doc {i} vec:", doc_vecs[i], len(doc_vecs[i]))