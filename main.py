from transformers import AutoTokenizer, pipeline
import torch
import logging
import warnings
from unsloth import FastLanguageModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

# Silence specific transformers warnings that use the logging module
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# Keep warnings filter just in case
warnings.filterwarnings("ignore", message=".*generation_config.*")
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")


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
    tokenizer.clean_up_tokenization_spaces = False

    FastLanguageModel.for_inference(model)

    print(f"✅ {model_name} and tokenizer loaded successfully")
    return model, tokenizer


def load_embed_model(model_name):
    model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print(f"✅ {model_name} loaded successfully")
    return model


def create_langchain_llm(model, tokenizer, max_new_tokens=512):
    """
    Creates a text-generation pipeline and converts it into a LangChain LLM object.
    """
    # Update the model's generation config
    model.generation_config.update(
        max_new_tokens=max_new_tokens,
        temperature=0.1,
        do_sample=True,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        max_length=None,
    )

    # Create the pipeline
    text_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
    )

    # Enforce the tokenizer setting on the pipeline object itself
    if hasattr(text_gen, "tokenizer") and text_gen.tokenizer is not None:
        text_gen.tokenizer.clean_up_tokenization_spaces = False

    # Convert to LangChain LLM
    llm = HuggingFacePipeline(pipeline=text_gen)

    print("✅ LangChain LLM pipeline is ready.")
    return llm


def generate_response(prompt, llm, tokenizer):
    """
    Formats the prompt into a chat template and generates a response using the LLM.
    """
    messages = [{"role": "user", "content": prompt}]

    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    response = llm.invoke(formatted_prompt)
    clean_response = response.replace(formatted_prompt, "").strip()

    return clean_response


def build_rag_prompt(context, question):
    template = """
    You are a helpful and knowledgeable AI assistant. Use only the information retrieved from the documents to answer the user's question in Persian (Farsi). If the answer is not found in the retrieved context, respond with: "متاسفانه اطلاعاتی در این مورد ندارم." Do not use your own knowledge beyond the provided context. Be accurate, clear, and polite. Never mention the documents or the retrieval process in your response. Just respond naturally in Persian.
    
Context:
{context}

Question:
{question}

Answer:
"""

    prompt = PromptTemplate.from_template(template)

    # Fill the template with actual values
    formatted_prompt = prompt.format(context=context, question=question)

    return formatted_prompt


if __name__ == "__main__":
    MODEL_NAME = "unsloth/gemma-3-4b-it-unsloth-bnb-4bit"
    EMBEDDING_MODEL_NAME = "PartAI/Tooka-SBERT-V2-Large"

    base_model, base_tokenizer = load_model_and_tokenizer(MODEL_NAME)
    embed_model = load_embed_model(EMBEDDING_MODEL_NAME)

    test_prompt = "Hello! What's your name?"
    langchain_framework = create_langchain_llm(base_model, base_tokenizer)

    # print("\n--- Generating Answer ---")
    # answer = generate_response(test_prompt, langchain_framework, base_tokenizer)
    # print(answer)

    sample_context = (
        "موتور ۴ لیتری شش سیلندر تخت تنفس طبیعی با ۵۲۵ اسب بخار قدرت و ۴۶۵ نیوتنمتر گشتاور"
    )
    sample_question = "اسم ماشین چیه؟"

    # 1. Build the prompt
    final_prompt = build_rag_prompt(context=sample_context, question=sample_question)

    print("--- Generated Prompt ---")
    print(final_prompt)
    print("------------------------")
    answer = generate_response(final_prompt, langchain_framework, base_tokenizer)
    print(answer)
