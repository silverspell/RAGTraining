from qdrant_client import QdrantClient, models
from FlagEmbedding import BGEM3FlagModel

# 1. Model + DB
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
client = QdrantClient(":memory:") # In-memory Qdrant DB for training

collection_name = "enterprise_rag"

# 2. Hem Dense hem Sparse vektörleri destekleyen collection oluşturma
client.create_collection(
    collection_name=collection_name,
    vectors_config= {
        "dense": models.VectorParams(size=1024, distance=models.Distance.COSINE),
    },
    sparse_vectors_config= {
        "sparse": models.SparseVectorParams()
    }
)

docs = [
    "Yıllık izin politikası: 5 yıl üzeri çalışanlar 20 iş günü izin kullanır. Kod: HR-2024-VAC.",
    "Çalışanlar, izin taleplerini en az 2 hafta önceden İnsan Kaynakları departmanına bildirmelidir.",
    "Acil durum izinleri, yöneticinin onayı ile verilebilir ve bu tür izinler yıllık izinden düşülmez."
]

for idx, doc in enumerate(docs):
    embeddings = model.encode([doc], return_dense=True, return_sparse=True)
    dense_vec = embeddings["dense_vecs"][0].tolist()  # İlk dokümanın dense vektörü
    # Sparse vektörleri Qdrant formatına dönüştürme
    sparse_vec = models.SparseVector(
        indices = list(map(int, embeddings["lexical_weights"][0].keys())),
        values = list(embeddings["lexical_weights"][0].values())
    )

    # 3. Dokümanı Qdrant'a ekleme
    client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=idx,
                vector = {
                    "dense": dense_vec,
                    "sparse": sparse_vec
                },
                payload = {
                    "text": doc
                }
            )
        ]
    )

# 4. Hibrit arama (Reciprocal Rank Fusion)
query = "Çalışanlar kaç gün yıllık izin kullanabilir? (Kod: HR-2024-VAC)"
query_embeddings = model.encode([query], return_dense=True, return_sparse=True)

search_result = client.query_points(
    collection_name=collection_name,
    prefetch=[
        models.Prefetch(query=query_embeddings["dense_vecs"][0].tolist(), using="dense", limit=5),
        models.Prefetch(
            query=models.SparseVector(
                indices=list(map(int, query_embeddings["lexical_weights"][0].keys())),
                values=list(query_embeddings["lexical_weights"][0].values())
            ),
            using="sparse",
            limit=5
        )
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF)
)

for res in search_result.points:
    print(f"Sonuç: {res.payload['text']} (Skor: {res.score})")
