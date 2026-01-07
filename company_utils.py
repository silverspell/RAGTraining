from ast import Tuple
import numpy as np
from typing import Any
from FlagEmbedding import BGEM3FlagModel


def load_text(txt_path: str)->str:
    pass

def create_text_splitter(
        docs: list,
        *,
        encoding_name: str = "o200k_base",
        chunk_size: int = 600,
        chunk_overlap: int = 80
)->Tuple[Any, Any]:
    pass

def chunk_documents(
        docs: list,
        *,
        encoding_name: str = "o200k_base",
        chunk_size: int = 600,
        chunk_overlap: int = 80,
)->list:
    pass

def create_embedding(
        texts: list,
        *,
        model_name: str = "BAAI/bge-m3",
        use_fp16: bool = True,
        batch_size: int = 32,
        max_length: int = 1024,
        use_dense: bool = True,
        use_sparse: bool = False,
)->Any:
    """
    Dikkat: dense vector key: dense_vecs
    sparse vector key: sparse_vecs
    olarak dönecek.
    """
    pass


def ask_gemini(query: str, model_name: str = "gemini-2.5-flash") -> str:
    pass


## Similarity Funcs

def dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def l2_norm(a: np.ndarray) -> float:
    return float(np.linalg.norm(a))


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    denom = max(l2_norm(a) * l2_norm(b), eps)
    return dot(a, b) / denom


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def l2_normalize(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(a)
    return a / max(n, eps)