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

from utils import CorpusRow, load_corpus, cosine_similarity, embed_query_bge_m3, ask_gemini

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()

load_dotenv()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", type=str, required=True)
    ap.add_argument("--source_id", type=str, default=None)

    args = ap.parse_args()
    
    response = ask_gemini(args.query, args.source_id)
    print(f"Gemini response: {response}")
    


if __name__ == "__main__":
    main()