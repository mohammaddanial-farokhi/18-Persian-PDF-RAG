# ==============================================================================
# IMPORTS
# ==============================================================================
import warnings
import logging
from typing import List, Optional, Dict

import torch
import unsloth
from transformers import pipeline
from unsloth import FastLanguageModel

# LangChain Imports
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch

# ==============================================================================
# CONFIGURATION & WARNINGS SUPPRESSION
# ==============================================================================
# Silence specific transformers warnings that use the logging module
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

warnings.filterwarnings("ignore", message=".*generation_config.*")
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")


# ==============================================================================
# 1. MODEL LOADING & SETUP
# ==============================================================================
def load_model_and_tokenizer(model_name: str, max_seq_length: int = 2048):
    """Loads a 4-bit quantized model using Unsloth for fast RAG inference."""
    print(f"⏳ Loading LLM: {model_name}")

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
    print(f"✅ LLM loaded successfully.")
    return model, tokenizer


def load_embed_model(model_name: str):
    """Loads the embedding model for creating vector representations."""
    print(f"⏳ Loading Embedding Model: {model_name}")
    model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print(f"✅ Embedding model loaded successfully.")
    return model


def create_langchain_llm(model, tokenizer, max_new_tokens: int = 512):
    """Creates a text-generation pipeline and converts it into a LangChain LLM."""
    model.generation_config.update(
        max_new_tokens=max_new_tokens,
        temperature=0.1,
        do_sample=True,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        max_length=None,
    )

    text_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
    )

    if hasattr(text_gen, "tokenizer") and text_gen.tokenizer is not None:
        text_gen.tokenizer.clean_up_tokenization_spaces = False

    llm = HuggingFacePipeline(pipeline=text_gen)
    print("✅ LangChain LLM pipeline is ready.")
    return llm


# ==============================================================================
# 2. DOCUMENT PROCESSING (PDF to Vector Store)
# ==============================================================================
def load_pdf(file_path: str) -> List[Document]:
    """Loads a PDF file and returns a list of Document objects."""
    print(f"⏳ Loading PDF: {file_path}")
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        print(f"✅ Successfully loaded {len(pages)} pages.")
        return pages
    except Exception as e:
        print(f"❌ Error loading PDF: {e}")
        return []


def split_documents_into_chunks(
    documents: List[Document], chunk_size: int = 1024, chunk_overlap: int = 256
) -> List[Document]:
    """Splits documents into smaller text chunks for embedding."""
    print(f"⏳ Splitting documents (chunk_size={chunk_size}, overlap={chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Successfully split into {len(chunks)} chunks.")
    return chunks


def create_vector_store(documents: List[Document], embedding_model: Embeddings):
    """Creates an in-memory vector store from document chunks."""
    print(f"⏳ Creating vector store for {len(documents)} chunks...")
    try:
        vectorstore = DocArrayInMemorySearch.from_documents(
            documents, embedding=embedding_model
        )
        print("✅ Vector store created successfully.")
        return vectorstore
    except Exception as e:
        print(f"❌ Error creating vector store: {e}")
        return None


# ==============================================================================
# 3. RAG LOGIC (Prompting & Generation)
# ==============================================================================
def build_rag_prompt(context: str, question: str) -> str:
    """Creates and formats a RAG prompt using the provided context and question."""
    template = """
    You are a helpful and knowledgeable AI assistant. Use only the information retrieved from the documents to answer the user's question in Persian (Farsi). 
    If the answer is not found in the retrieved context, respond with: "متاسفانه اطلاعاتی در این مورد ندارم." 
    Do not use your own knowledge beyond the provided context. Be accurate, clear, and polite. 
    Never mention the documents or the retrieval process in your response. Just respond naturally in Persian.
    
    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate.from_template(template)
    return prompt.format(context=context, question=question)


def generate_response(prompt: str, llm, tokenizer) -> str:
    """Formats the prompt into a chat template and generates a response using the LLM."""
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    response = llm.invoke(formatted_prompt)
    return response.replace(formatted_prompt, "").strip()


def ask_multiple_questions(
    questions: List[str], retriever, llm, tokenizer
) -> List[Dict[str, str]]:
    """Takes a list of questions, retrieves context, and generates answers."""
    results = []
    for question in questions:
        print(f"\n❓ Question: {question}")
        retrieved_docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        final_prompt = build_rag_prompt(context=context, question=question)

        print("🧠 Thinking...")
        answer = generate_response(final_prompt, llm, tokenizer)
        print(f"🤖 Answer: {answer}")
        print("-" * 60)

        results.append({"question": question, "answer": answer})
    return results


# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # --- Configuration ---
    MODEL_NAME = "unsloth/gemma-3-4b-it-unsloth-bnb-4bit"
    EMBEDDING_MODEL_NAME = "PartAI/Tooka-SBERT-V2-Large"
    PDF_PATH = "test_sample.pdf"

    # --- Step 1: Load Models ---
    base_model, base_tokenizer = load_model_and_tokenizer(MODEL_NAME)
    embed_model = load_embed_model(EMBEDDING_MODEL_NAME)
    langchain_llm = create_langchain_llm(base_model, base_tokenizer)

    # --- Step 2: Process PDF ---
    documents = load_pdf(PDF_PATH)

    if not documents:
        print("❌ Exiting because PDF could not be loaded.")
        exit()

    chunks = split_documents_into_chunks(documents)
    vectorstore = create_vector_store(chunks, embedding_model=embed_model)

    # --- Step 3: Ask Questions ---
    if vectorstore:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

        my_questions = [
            "اندازه‌ی پیش فرض Tab چند سانتی متر می باشد؟",
            "پسوند فایل های الگو در نرم افزار Word 2019 چیست؟",
        ]

        answers = ask_multiple_questions(
            questions=my_questions,
            retriever=retriever,
            llm=langchain_llm,
            tokenizer=base_tokenizer,
        )
