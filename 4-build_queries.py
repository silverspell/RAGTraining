import json
from pathlib import Path

def write_jsonl(rows, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    queries = [
        {
            "id": "q_000001",
            "query": "İade süresi kaç gün?",
            "metadata": {"lang": "tr", "domain": "ecommerce", "difficulty": "easy"},
            "ground_truth": {
                "source_ids": ["policy_returns_tr"],
                "chunk_ids": [],
                "answer_text": "Teslimat tarihinden itibaren 14 gün."
            }
        },
        {
            "id": "q_000002",
            "query": "Kargo takip numarasını nereden bulurum?",
            "metadata": {"lang": "tr", "domain": "ecommerce", "difficulty": "easy"},
            "ground_truth": {
                "source_ids": ["policy_returns_tr"],
                "chunk_ids": [],
                "answer_text": "Sipariş sayfasında yer alır."
            }
        },
        {
            "id": "q_000003",
            "query": "Ücret iadesi kaç günde yapılır?",
            "metadata": {"lang": "tr", "domain": "ecommerce", "difficulty": "medium"},
            "ground_truth": {
                "source_ids": ["policy_returns_tr"],
                "chunk_ids": [],
                "answer_text": "İade onayından sonra 3-10 iş günü."
            }
        },
        {
            "id": "q_000004",
            "query": "Kişisel verilerim hangi amaçlarla işlenir?",
            "metadata": {"lang": "tr", "domain": "ecommerce", "difficulty": "easy"},
            "ground_truth": {
                "source_ids": ["policy_privacy_tr"],
                "chunk_ids": [],
                "answer_text": "Hizmet sunumu ve müşteri desteği amaçlarıyla işlenir."
            }
            },
            {
            "id": "q_000005",
            "query": "Verilerim üçüncü taraflarla paylaşılır mı?",
            "metadata": {"lang": "tr", "domain": "ecommerce", "difficulty": "medium"},
            "ground_truth": {
                "source_ids": ["policy_privacy_tr"],
                "chunk_ids": [],
                "answer_text": "Yasal yükümlülükler veya açık rızanız olmaksızın paylaşılmaz."
            }
            },
            {
            "id": "q_000006",
            "query": "Çerezleri nasıl yönetebilirim?",
            "metadata": {"lang": "tr", "domain": "ecommerce", "difficulty": "easy"},
            "ground_truth": {
                "source_ids": ["policy_privacy_tr"],
                "chunk_ids": [],
                "answer_text": "Tarayıcı ayarlarınızdan çerezleri yönetebilirsiniz."
            }
            },

    ]

    out_path = "data/queries.jsonl"
    write_jsonl(queries, out_path)
    print(f"Wrote: {out_path} (rows={len(queries)})")
