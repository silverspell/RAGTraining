from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from FlagEmbedding import BGEM3FlagModel


# -----------------------------
# Data models
# -----------------------------
@dataclass
class Doc:
    source_id: str
    text: str
    title: Optional[str] = None
    url: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


@dataclass
class Chunk:
    id: str
    text: str
    metadata: Dict[str, Any]


@dataclass
class CorpusItem:
    id: str
    text: str
    metadata: Dict[str, Any]
    vector: List[float] 


# -----------------------------
# Utilities
# -----------------------------
def clean_text(text: str) -> str:
    """
    Eğitim corpusu için temizlik:
    - fazla boşluklar
    - satır sonu/paragraph normalize
    - PDF'den gelen tire ile bölünmüş kelime (ör: "do-\nküman")
    """
    if not text:
        return ""

    # Hyphenation fix: "do-\nküman" -> "doküman"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Normalize newlines: 3+ newline -> 2 newline
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Trim spaces each line
    text = "\n".join(line.strip() for line in text.splitlines())

    # Collapse multiple spaces
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def stable_chunk_id(source_id: str, chunk_index: int, chunk_text: str) -> str:
    h = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()[:16]
    return f"{source_id}:{chunk_index}:{h}"


def build_splitter(
    *,
    encoding_name: str = "o200k_base",
    chunk_size: int = 700,       # token hedefi
    chunk_overlap: int = 100,    # token overlap
) -> tuple[RecursiveCharacterTextSplitter, Any]:
    enc = tiktoken.get_encoding(encoding_name)

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=encoding_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ": ",
            ", ",
            " ",
            "",
        ],
        add_start_index=True,  # start_index metadata
    )
    return splitter, enc


def chunk_docs(
    docs: Iterable[Doc],
    *,
    encoding_name: str = "o200k_base",
    chunk_size: int = 700,
    chunk_overlap: int = 100,
) -> List[Chunk]:
    splitter, enc = build_splitter(
        encoding_name=encoding_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    out: List[Chunk] = []
    for d in docs:
        text = clean_text(d.text)
        if not text:
            continue

        base_meta: Dict[str, Any] = {
            "source_id": d.source_id,
            "title": d.title,
            "url": d.url,
        }
        if d.extra:
            base_meta.update(d.extra)

        # create_documents -> her chunk için start_index ekler
        pieces = splitter.create_documents([text], metadatas=[base_meta])

        for i, p in enumerate(pieces):
            chunk_text = p.page_content.strip()
            if not chunk_text:
                continue

            token_count = len(enc.encode(chunk_text))

            md = dict(p.metadata or {})
            md.update({
                "chunk_index": i,
                "token_count": token_count,
            })

            cid = stable_chunk_id(d.source_id, i, chunk_text)
            out.append(Chunk(id=cid, text=chunk_text, metadata=md))

    return out


def embed_chunks_bge_m3(
    chunks: List[Chunk],
    *,
    model_name: str = "BAAI/bge-m3",
    use_fp16: bool = True,
    max_length: int = 1024,
    batch_size: int = 32,
) -> List[CorpusItem]:
    model = BGEM3FlagModel(model_name, use_fp16=use_fp16)

    texts = [c.text for c in chunks]
    out = model.encode(
        texts,
        batch_size=batch_size,
        max_length=max_length,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )

    vecs = out["dense_vecs"]  # numpy array (N, 1024) ya da list
    vecs = np.asarray(vecs)

    corpus: List[CorpusItem] = []
    for c, v in zip(chunks, vecs):
        corpus.append(CorpusItem(
            id=c.id,
            text=c.text,
            metadata=c.metadata,
            vector=v.astype(np.float32).tolist()
        ))
    return corpus


def write_jsonl(items: List[CorpusItem], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps({
                "id": it.id,
                "text": it.text,
                "metadata": it.metadata,
                "vector": it.vector,
            }, ensure_ascii=False) + "\n")


def print_stats(chunks: List[Chunk]) -> None:
    if not chunks:
        print("No chunks.")
        return
    toks = [c.metadata["token_count"] for c in chunks]
    print(f"chunks: {len(chunks)}")
    print(f"min/avg/max token: {min(toks)} / {sum(toks)/len(toks):.1f} / {max(toks)}")
    # en uzun 3 chunk preview
    longest = sorted(chunks, key=lambda c: c.metadata["token_count"], reverse=True)[:3]
    print("\nLongest previews:")
    for c in longest:
        prev = c.text[:140].replace("\n", " ")
        print(f"- {c.id} | {c.metadata['token_count']} tok | {prev}...")


# -----------------------------
# Demo / entrypoint
# -----------------------------
if __name__ == "__main__":
    # 1) Eğitim dokümanları 
    docs = [
        Doc(
            source_id="policy_returns_tr",
            title="İade Politikası",
            url="https://example.com/iade",
            text=(
                "İade politikamız aşağıdaki gibidir.\n\n"
                "1) Teslimat tarihinden itibaren 14 gün içinde iade talebi oluşturabilirsiniz. "
                "Ürün kullanılmamış olmalıdır.\n\n"
                "2) Kargo takip numarası sipariş sayfasında yer alır. "
                "İade kargo bedeli kampanya dönemlerinde ücretsiz olabilir.\n\n"
                "3) İade onayı sonrasında ücret iadesi 3-10 iş günü içinde gerçekleştirilir.\n"
            ),
            extra={"doc_type": "policy", "lang": "tr"},
        ),
        Doc(
            source_id="policy_privacy_tr",
            title="Gizlilik Politikası",
            url="https://example.com/gizlilik",
            text=(
                "Gizlilik politikamız aşağıdaki gibidir.\n\n"
                "1) Kişisel verileriniz, hizmet sunumu ve müşteri desteği amaçlarıyla işlenir.\n\n"
                "2) Verileriniz, yasal yükümlülükler veya açık rızanız olmaksızın üçüncü taraflarla paylaşılmaz.\n\n"
                "3) Çerezler (cookies), deneyimi iyileştirmek ve analiz yapmak için kullanılabilir. "
                "Tarayıcı ayarlarınızdan çerezleri yönetebilirsiniz.\n\n"
                "4) Veri saklama süresi, ilgili mevzuata uygun şekilde belirlenir; gerekli olmadığında veriler silinir veya anonimleştirilir.\n"
            ),
            extra={"doc_type": "policy", "lang": "tr"},
        ),
    ]

    # 2) Chunk
    chunks = chunk_docs(
        docs,
        encoding_name="o200k_base",
        chunk_size=30,
        chunk_overlap=5,
    )
    print_stats(chunks)

    # 3) Embed
    corpus_items = embed_chunks_bge_m3(
        chunks,
        model_name="BAAI/bge-m3",
        use_fp16=True,
        max_length=1024,
        batch_size=32,
    )

    # 4) Export (eğitim corpus formatı)
    out_path = "data/corpus_tr_bge_m3.jsonl"
    write_jsonl(corpus_items, out_path)
    print(f"\nWrote: {out_path} (items={len(corpus_items)})")

    # 5) Sanity: ilk item metadata + vector dim
    first = corpus_items[0]
    print("\n--- First item ---")
    print("id:", first.id)
    print("metadata:", first.metadata)
    print("vector_dim:", len(first.vector))
    print("text_preview:", first.text[:200])
