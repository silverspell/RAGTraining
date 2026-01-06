import os
from qdrant_client import QdrantClient, models
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from google import genai
from dotenv import load_dotenv
# Ortam değişkenlerini yükle
load_dotenv()

# 1. MODELLERİN VE İSTEMCİLERİN HAZIRLANMASI
embedding_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
reranker_model = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

# Gemini Client (API Key'inizi buraya ekleyin veya environment'a tanımlayın)
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Qdrant (Hızlı demo için bellek üzerinde)
q_client = QdrantClient(":memory:")
COLLECTION_NAME = "enterprise_knowledge"

# 2. QDRANT KOLEKSİYON YAPILANDIRMASI (DENSE + SPARSE)
q_client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={"dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)},
    sparse_vectors_config={"sparse": models.SparseVectorParams()}
)

# 3. INGESTION (VERİ YÜKLEME) FONKSİYONU
def ingest_document(doc_id, text, metadata):
    outputs = embedding_model.encode([text], return_dense=True, return_sparse=True)
    
    dense_vec = outputs["dense_vecs"][0].tolist()
    sparse_dict = outputs["lexical_weights"][0]
    sparse_vec = models.SparseVector(
        indices=[int(i) for i in sparse_dict.keys()],
        values=list(sparse_dict.values())
    )
    
    q_client.upsert(
        collection_name=COLLECTION_NAME,
        points=[models.PointStruct(
            id=doc_id,
            vector={"dense": dense_vec, "sparse": sparse_vec},
            payload={"text": text, **metadata}
        )]
    )

# Örnek Veri Ekleme
ingest_document(1, "Şirket sağlık sigortası, yatarak tedavilerde %100 kapsam sağlar.", {"topic": "sigorta"})
ingest_document(2, "Diş tedavileri yıllık 5000 TL limit ile sınırlıdır.", {"topic": "sigorta"})
ingest_document(3, "Yıllık izinler İK sistemi üzerinden Ocak ayında planlanmalıdır.", {"topic": "ik"})
ingest_document(4, "Çalışanlar, yıllık izinlerini en az 2 hafta önceden bildirmelidir.", {"topic": "ik"})
ingest_document(5, "Emeklilik planları için şirket katkısı %5'tir.", {"topic": "emeklilik"})
ingest_document(6, "Emeklilik yaş sınırı 60'tır.", {"topic": "emeklilik"})


# 4. RETRIEVAL + RERANKING + GENERATION AKIŞI
def ask_question(query):
    print(f"\nSoru: {query}")
    
    # A. Hibrit Arama (Dense + Sparse)
    query_outputs = embedding_model.encode([query], return_dense=True, return_sparse=True)
    
    search_results = q_client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=query_outputs["dense_vecs"][0].tolist(), using="dense", limit=20),
            models.Prefetch(
                query=models.SparseVector(
                    indices=[int(i) for i in query_outputs["lexical_weights"][0].keys()],
                    values=list(query_outputs["lexical_weights"][0].values())
                ),
                using="sparse",
                limit=20
            )
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF)
    )
    
    candidates = [res.payload["text"] for res in search_results.points]
    
    print("----------RERANKER ÖNCESİ----------")
    for i, candidate in enumerate(candidates, 1):
        print(f"{i}. {candidate}")

    # B. Reranking (En iyi sonuçları seçme)
    rerank_pairs = [[query, doc] for doc in candidates]
    rerank_scores = reranker_model.compute_score(rerank_pairs)
    
    # Skorlara göre sırala
    ranked_docs = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)

    print("----------RERANKER SONRASI----------")
    for i, (doc, score) in enumerate(ranked_docs, 1):
        print(f"{i}. (Skor: {score:.4f}) {doc}")

    context = "\n\n".join([doc for doc, score in ranked_docs[:3]])
    
    # C. Generation (Gemini)
    final_prompt = f"""
        Aşağıdaki "KAYNAKLAR" bölümünde verilen bilgileri kullanarak, "SORU"yu yanıtla.
        Eğer kaynaklar soruyu yanıtlamak için yeterli değilse, "Bilmiyorum" diye cevap ver.

        KAYNAKLAR:
        {context}

        SORU:
        {query}

        Cevap:
    """
    response = client.models.generate_content(model="gemini-2.5-flash-lite", contents=final_prompt)
    
    return response.text, ranked_docs

# ÇALIŞTIR
answer, sources = ask_question("Sağlık sigortam diş operasyonlarını ne kadar kapsıyor?")
print(f"\nYanıt:\n{answer}")
print("\nKullanılan Kaynaklar (Reranked):", [s[0] for s in sources])