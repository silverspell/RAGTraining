from dotenv import load_dotenv
import argparse
from utils import ask_gemini, ask_gemini_cli

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
    
    response = ask_gemini_cli(args.query, args.source_id)
    print("-----------------------------------------")
    print(f"Gemini response: {response}")
    


if __name__ == "__main__":
    main()