import streamlit as st
import pandas as pd
import plotly.express as px
from datasets import Dataset
import os
from dotenv import load_dotenv

import asyncio
import threading
import concurrent.futures

from openai import OpenAI, AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextRecall,
    ContextPrecision,
)

load_dotenv()

# -----------------------------
# ✅ UVLOOP SAFE ASYNC RUNNER
# -----------------------------
def _get_bg_loop():
    """
    Tek bir background event-loop thread'i oluşturur ve session_state'te saklar.
    uvloop/nest_asyncio ile uğraşmayız.
    """
    if "_bg_loop" in st.session_state and st.session_state["_bg_loop"] is not None:
        return st.session_state["_bg_loop"]

    ready = threading.Event()
    holder = {}

    def _runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        holder["loop"] = loop
        ready.set()
        loop.run_forever()

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    ready.wait()

    st.session_state["_bg_loop"] = holder["loop"]
    st.session_state["_bg_thread"] = t
    return holder["loop"]


def run_async(coro):
    """
    Her koşulda coroutine'i güvenli çalıştırır:
    - Eğer mevcut thread'de running loop yoksa: asyncio.run
    - Varsa (uvloop dahil): background loop'a submit eder
    """
    try:
        asyncio.get_running_loop()
        bg = _get_bg_loop()
        fut = asyncio.run_coroutine_threadsafe(coro, bg)
        return fut.result()
    except RuntimeError:
        return asyncio.run(coro)


# ---------------- UI ----------------
st.set_page_config(page_title="RAG Quality Dashboard (OpenAI)", layout="wide")
st.title("📊 RAG Kalite Ölçüm Paneli (OpenAI)")
st.markdown("Bu panel, **OpenAI** modellerini 'Hakem' (Judge) olarak kullanarak RAG sisteminizi analiz eder.")

with st.sidebar:
    st.header("⚙️ Yapılandırma")
    openai_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY") or "")
    model_choice = st.selectbox("Değerlendirici Model", ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"])
    translate_instructions = st.checkbox("Prompt instruction'larını da Türkçeleştir", value=True)
    st.info("💡 İpucu: Faithfulness için gpt-4o daha stabil sonuç verir.")

# Örnek Veri
if "eval_data" not in st.session_state:
    st.session_state.eval_data = {
        'question': [
            "Yıllık izin süresi kaç gündür?",
            "Şirket arabası kimlere verilir?",
            "Eğitim bütçesi ne kadar?"
        ],
        'answer': [
            "14 veya 20 gündür.",
            "Sadece saha satış personeline verilir.",
            "Bütçe her yıl yönetim tarafından belirlenir."
        ],
        'contexts': [
            ["1-5 yıl arası 14 gün, 5 yıldan fazla olanlar 20 gün izin kullanır."],
            ["Saha operasyon ve satış temsilcileri şirket aracı hakkına sahiptir."],
            ["Eğitim departmanı yıllık bütçesini her Ocak ayında revize eder."]
        ],
        'ground_truth': [
            "1-5 yıl arası 14 gün, 5 yıl üstü 20 gündür.",
            "Satış ve saha ekipleri araç alabilir.",
            "Eğitim bütçesi yıllık olarak Ocak ayında belirlenir."
        ]
    }

st.subheader("📋 Değerlendirilecek Örnek Veri Seti")
st.dataframe(pd.DataFrame(st.session_state.eval_data), use_container_width=True)


# ---------------- RAGAS helpers ----------------
def build_llm_and_embeddings(openai_key: str, model_choice: str):
    """
    Collections metrikleri modern InstructorLLM ister.
    Embeddings tarafı async çalışabildiği için AsyncOpenAI client veriyoruz.
    """
    async_client = AsyncOpenAI(api_key=openai_key)

    llm = llm_factory(model_choice, client=async_client)

    # ✅ embeddings de async client ile
    embeddings = embedding_factory(
        "openai",
        model="text-embedding-3-small",
        client=async_client,
        interface="modern",
    )

    return llm, embeddings


async def adapt_metric_prompts_to_tr(metric, adapt_instruction: bool):
    """
    Metric içindeki prompt alanlarını TR'ye adapte eder.
    """
    for attr in ["prompt", "statement_generator_prompt", "nli_statement_prompt"]:
        if hasattr(metric, attr):
            p = getattr(metric, attr)
            if p is not None and hasattr(p, "adapt"):
                setattr(
                    metric,
                    attr,
                    await p.adapt(
                        target_language="turkish",
                        llm=metric.llm,
                        adapt_instruction=adapt_instruction
                    )
                )
    return metric


async def score_metric_batch(metric, inputs):
    """
    batch_ascore varsa kullanır.
    Yoksa tek tek ascore fallback.
    """
    if hasattr(metric, "batch_ascore"):
        return await metric.batch_ascore(inputs)

    # fallback: tek tek
    out = []
    if hasattr(metric, "ascore"):
        for d in inputs:
            out.append(await metric.ascore(**d))
        return out

    # en son fallback (sync)
    return [metric.score(**d) for d in inputs]


async def evaluate_collections(df: pd.DataFrame, metrics: dict):
    """
    df kolonları: question, answer, contexts, ground_truth
    metrics: {"faithfulness": Faithfulness(...), ...}
    """
    faith_inputs = [
        {"user_input": r["question"], "response": r["answer"], "retrieved_contexts": r["contexts"]}
        for _, r in df.iterrows()
    ]
    ansrel_inputs = [
        {"user_input": r["question"], "response": r["answer"]}
        for _, r in df.iterrows()
    ]
    ctxrec_inputs = [
        {"user_input": r["question"], "retrieved_contexts": r["contexts"], "reference": r["ground_truth"]}
        for _, r in df.iterrows()
    ]
    ctxprec_inputs = [
        {"user_input": r["question"], "retrieved_contexts": r["contexts"], "reference": r["ground_truth"]}
        for _, r in df.iterrows()
    ]

    results = {}

    faith_res = await score_metric_batch(metrics["faithfulness"], faith_inputs)
    results["faithfulness"] = [float(getattr(x, "value", x)) for x in faith_res]

    ansrel_res = await score_metric_batch(metrics["answer_relevancy"], ansrel_inputs)
    results["answer_relevancy"] = [float(getattr(x, "value", x)) for x in ansrel_res]

    ctxrec_res = await score_metric_batch(metrics["context_recall"], ctxrec_inputs)
    results["context_recall"] = [float(getattr(x, "value", x)) for x in ctxrec_res]

    ctxprec_res = await score_metric_batch(metrics["context_precision"], ctxprec_inputs)
    results["context_precision"] = [float(getattr(x, "value", x)) for x in ctxprec_res]

    return pd.DataFrame(results)


# ---------------- Main action ----------------
if st.button("🚀 Değerlendirmeyi Başlat (OpenAI-Judge)"):
    if not openai_key:
        st.error("Lütfen bir OpenAI API Key girin!")
    else:
        with st.spinner("OpenAI modelleri analiz yapıyor... (TR prompt + collections scoring)"):
            try:
                df = pd.DataFrame(st.session_state.eval_data)

                cache_key = f"collections_metrics::{model_choice}::instr={int(translate_instructions)}"

                async def init_metrics():
                    llm, embeddings = build_llm_and_embeddings(openai_key, model_choice)

                    faith = Faithfulness(llm=llm)
                    ansrel = AnswerRelevancy(llm=llm, embeddings=embeddings)
                    ctxrec = ContextRecall(llm=llm)
                    ctxprec = ContextPrecision(llm=llm)

                    # TR prompt adapt
                    faith = await adapt_metric_prompts_to_tr(faith, translate_instructions)
                    ansrel = await adapt_metric_prompts_to_tr(ansrel, translate_instructions)
                    ctxrec = await adapt_metric_prompts_to_tr(ctxrec, translate_instructions)
                    ctxprec = await adapt_metric_prompts_to_tr(ctxprec, translate_instructions)

                    return {
                        "faithfulness": faith,
                        "answer_relevancy": ansrel,
                        "context_recall": ctxrec,
                        "context_precision": ctxprec,
                    }

                if cache_key not in st.session_state:
                    st.session_state[cache_key] = run_async(init_metrics())

                metrics = st.session_state[cache_key]

                # Evaluate
                res_df = run_async(evaluate_collections(df, metrics))

                st.success("Analiz Başarıyla Tamamlandı!")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Faithfulness", f"{res_df['faithfulness'].mean():.2f}")
                c2.metric("Answer Relevancy", f"{res_df['answer_relevancy'].mean():.2f}")
                c3.metric("Context Recall", f"{res_df['context_recall'].mean():.2f}")
                c4.metric("Context Precision", f"{res_df['context_precision'].mean():.2f}")

                plot_df = res_df[['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision']].mean().reset_index()
                plot_df.columns = ['Metric', 'Score']
                fig = px.bar(plot_df, x='Metric', y='Score', color='Metric', range_y=[0, 1], title="Ortalama Başarı Skorları")
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("🔍 Detaylı Analiz Tablosu")
                st.dataframe(res_df, use_container_width=True)

            except Exception as e:
                st.error(f"Hata: {str(e)}")
                st.caption("Eğer openai/ragas sürüm uyuşmazlığı varsa: `pip show ragas openai` çıktısını at, netleştireyim.")
