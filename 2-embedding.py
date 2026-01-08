from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True) # GPU varsa fp16 kullan

docs = [
    "İade politikamız: Teslimden itibaren 30 gün içinde kullanılmamış ürünleri iade edebilirsiniz.",
    "Kargo takip numarası sipariş sayfasında yer alır."
]

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

q_out = model.encode(
    ["İade süresi nedir?"],
    batch_size=1,
    max_length=256,
    return_dense=True,
    return_sparse=False,
    return_colbert_vecs=False
)

q_vec = q_out["dense_vecs"] # (1, 1024)
print(q_vec[0])