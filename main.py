from transformers import AutoTokenizer, pipeline
import torch
from unsloth import FastLanguageModel


def load_model_and_tokenizer(model_name: str, max_seq_length: int = 2048):
    """
    Loads a 4-bit quantized model using Unsloth for fast RAG inference.
    """
    print(f"Loading model with Unsloth: {model_name}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        local_files_only=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    FastLanguageModel.for_inference(model)

    print("✅ Model loaded and ready for RAG inference!")
    return model, tokenizer


if __name__ == "__main__":
    MODEL_NAME = "unsloth/gemma-3-4b-it-unsloth-bnb-4bit"

    base_model, base_tokenizer = load_model_and_tokenizer(MODEL_NAME)
