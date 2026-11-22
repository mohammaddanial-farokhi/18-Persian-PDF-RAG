import warnings
import logging
import torch
import gradio as gr

# Optional import for Hugging Face Spaces ZeroGPU
try:
    import spaces
except ImportError:

    class spaces:
        @staticmethod
        def GPU(func):
            return func


# Transformers & Unsloth
import unsloth
from transformers import pipeline
from unsloth import FastLanguageModel

# LangChain
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch

# ==============================================================================
# CONFIGURATION & WARNINGS SUPPRESSION
# ==============================================================================
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*generation_config.*")
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")

MODEL_NAME = "unsloth/gemma-3-4b-it-unsloth-bnb-4bit"
EMBEDDING_MODEL_NAME = "PartAI/Tooka-SBERT-V2-Large"

# ==============================================================================
# 1. MODEL INITIALIZATION
# ==============================================================================
print("⏳ Initializing Models (This might take a few minutes)...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=2048,
    dtype=torch.bfloat16,
    load_in_4bit=True,
    local_files_only=False,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.clean_up_tokenization_spaces = False
FastLanguageModel.for_inference(model)

model.generation_config.update(
    max_new_tokens=512,
    temperature=0.1,
    do_sample=True,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    max_length=None,
)
text_gen = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, return_full_text=False
)
if hasattr(text_gen, "tokenizer") and text_gen.tokenizer is not None:
    text_gen.tokenizer.clean_up_tokenization_spaces = False
llm = HuggingFacePipeline(pipeline=text_gen)

embed_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)
print("✅ All models loaded successfully!")


# ==============================================================================
# 2. CORE FUNCTIONS
# ==============================================================================
def process_pdf(file_path, progress=gr.Progress()):
    if not file_path:
        return None, "⚠️ لطفاً ابتدا یک فایل PDF آپلود کنید.", gr.update(interactive=False)

    try:
        progress(0.1, desc="در حال خواندن فایل PDF...")
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()

        progress(0.4, desc="در حال تکه‌تکه کردن متن...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600, chunk_overlap=100, length_function=len
        )
        chunks = text_splitter.split_documents(pages)

        progress(0.7, desc="در حال ساخت پایگاه داده برداری...")
        vectorstore = DocArrayInMemorySearch.from_documents(chunks, embedding=embed_model)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        progress(1.0, desc="پردازش کامل شد!")
        return (
            retriever,
            f"✅ فایل پردازش شد! ({len(pages)} صفحه، {len(chunks)} قطعه)",
            gr.update(interactive=True),
        )
    except Exception as e:
        return None, f"❌ خطا: {str(e)}", gr.update(interactive=False)


def build_rag_prompt(context: str, question: str) -> str:
    template = """
    You are a friendly, polite, and highly conversational Persian AI assistant.
    Your goal is to answer the user's question based ONLY on the provided Context, just like a human tutor explaining a concept.

    IMPORTANT INSTRUCTIONS:
    1. Read the Context carefully (it might contain mixed English/Persian text).
    2. Start your answer with a natural transition (e.g., "بر اساس اطلاعات موجود در سند شما، ...").
    3. Highlight the core answer using bold text (e.g., **Shift Cells Right**).
    4. DO NOT just output a single word or a short robotic phrase. Write a full, helpful paragraph.

    CRITICAL RULE: 
    NEVER end your answer with generic closing phrases like "امیدوارم این توضیح مفید بوده باشد", "امیدوارم کمک کرده باشم", or "اگر سوال دیگری دارید بپرسید". End your response immediately after the explanation. Just stop!

    ---
    EXAMPLE OF A GOOD RESPONSE:
    Context: "برای انتقال خانه‌ها به راست از Shift Cells Right استفاده کنید. این گزینه خانه‌های سمت راست را جابه‌جا می‌کند."
    Question: "کدام گزینه خانه‌ها را به راست می‌برد؟"
    Answer: "بر اساس اطلاعات موجود در سند شما، گزینه صحیح **Shift Cells Right** است. این گزینه به طور خاص خانه‌های جدول را به سمت راست جابه‌جا می‌کند."
    ---

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    return PromptTemplate.from_template(template).format(context=context, question=question)


@spaces.GPU
def answer_question(question: str, retriever):
    if not retriever:
        return "⚠️ لطفاً ابتدا یک فایل PDF آپلود و پردازش کنید.", ""
    if not question.strip():
        return "⚠️ لطفاً سوال خود را وارد کنید.", ""

    retrieved_docs = retriever.invoke(question)
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    reference_text = ""
    for i, doc in enumerate(retrieved_docs, 1):
        page = doc.metadata.get("page", 0) + 1
        reference_text += (
            f"📄 **بخش {i} (صفحه {page}):**\n{doc.page_content.strip()}\n\n{'-'*40}\n"
        )

    final_prompt = build_rag_prompt(context=context_text, question=question)
    messages = [{"role": "user", "content": final_prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    response = llm.invoke(formatted_prompt)
    clean_response = response.replace(formatted_prompt, "").strip()

    return clean_response, reference_text


# ==============================================================================
# 3. GRADIO UI (STRICT SINGLE-SCREEN CSS)
# ==============================================================================

custom_css = """
@import url('https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css');

/* ------------------------------------------------
   GLOBAL FONT & ALIGNMENT
------------------------------------------------ */

* {
    font-family: 'Vazirmatn', sans-serif !important;
}

html, body {
    height: 100vh !important;
    overflow: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
}

body {
    display: flex !important;
    justify-content: center !important;
}

gradio-app {
    width: 100% !important;
}

body, .gradio-container {
    font-family: 'Vazirmatn', sans-serif !important;
    direction: rtl !important;
    text-align: center !important;
    background-color: #f8fafc;
}

.gradio-container {
    width: 100% !important;
    max-width: 1100px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    height: 100vh !important;
    padding: 10px !important;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

/* ------------------------------------------------
   CENTER ALL COMPONENTS
------------------------------------------------ */

.gradio-container,
.gradio-container .block,
.gradio-container .form,
.gradio-container .wrap,
.gradio-container label,
.gradio-container .label-wrap,
.gradio-container .prose,
.gradio-container p,
.gradio-container h1,
.gradio-container h2,
.gradio-container h3 {
    text-align: center !important;
}

/* ------------------------------------------------
   PERSIAN & ENGLISH TEXT DIRECTION
------------------------------------------------ */

textarea,
input,
input[type="text"],
input[type="number"] {
    direction: rtl !important;
    text-align: center !important;
    unicode-bidi: plaintext !important;
    font-family: 'Vazirmatn', sans-serif !important;
}

textarea::placeholder,
input::placeholder {
    direction: rtl !important;
    text-align: center !important;
    unicode-bidi: plaintext !important;
    font-family: 'Vazirmatn', sans-serif !important;
}

/* Output and reference text */
textarea,
.markdown-text,
.prose,
.prose p,
.prose li,
.output-markdown,
[data-testid="textbox"] textarea {
    direction: rtl !important;
    text-align: center !important;
    unicode-bidi: plaintext !important;
    font-family: 'Vazirmatn', sans-serif !important;
    line-height: 2 !important;
}

/* English text and technical terms */
code,
pre {
    direction: ltr !important;
    unicode-bidi: isolate !important;
    font-family: Arial, sans-serif !important;
}

/* ------------------------------------------------
   COMPACT LAYOUT - PRESERVED
------------------------------------------------ */

.gap {
    gap: 5px !important;
}

.block {
    padding: 2px !important;
}

/* ------------------------------------------------
   FILE UPLOAD - PRESERVED
------------------------------------------------ */

div[data-testid="block-file"] {
    min-height: 50px !important;
    max-height: 50px !important;
    height: 50px !important;
    overflow: hidden !important;
}

div[data-testid="block-file"] > div {
    min-height: 50px !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

div[data-testid="block-file"] svg {
    display: none !important;
}

div[data-testid="block-file"] span {
    display: none !important;
}

/* ------------------------------------------------
   TEXTBOX HEIGHTS - PRESERVED
------------------------------------------------ */

textarea {
    font-size: 0.95rem !important;
    height: 120px !important;
    max-height: 120px !important;
    overflow-y: auto !important;
}

.ref-box textarea {
    height: 80px !important;
    max-height: 80px !important;
    font-size: 0.85rem !important;
    background-color: #f1f5f9 !important;
}

/* ------------------------------------------------
   BUTTONS - PRESERVED
------------------------------------------------ */

button.primary {
    background: linear-gradient(
        135deg,
        #10b981 0%,
        #059669 100%
    ) !important;
    color: white !important;
    border: none !important;
}

button.secondary {
    background: #334155 !important;
    color: white !important;
}

/* ------------------------------------------------
   HEADINGS
------------------------------------------------ */

h1 {
    margin: 0 !important;
    font-size: 1.8rem !important;
    color: #1e293b;
    text-align: center !important;
}

p {
    margin: 0 !important;
    font-size: 1.1rem !important;
    color: #475569;
    text-align: center !important;
}

/* Model and Embedding Information */
.gradio-container div[style*="font-size: 0.75em"] {
    font-size: 1rem !important;
}

/* ------------------------------------------------
   LABELS & STATUS
------------------------------------------------ */

label,
span,
.gradio-container .wrap-inner {
    font-family: 'Vazirmatn', sans-serif !important;
}

.gradio-container input,
.gradio-container textarea {
    text-align: center !important;
}

/* ------------------------------------------------
   REFERENCE ACCORDION TITLE
------------------------------------------------ */

.gradio-container [data-testid="accordion"] > button {
    direction: rtl !important;
    text-align: right !important;
    justify-content: flex-start !important;
}

.gradio-container [data-testid="accordion"] > button span {
    text-align: right !important;
    direction: rtl !important;
}

"""


with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="emerald", secondary_hue="slate", font=gr.themes.GoogleFont("Vazirmatn")
    ),
    css=custom_css,
    title="Persian RAG Assistant",
) as demo:

    # --- Ultra Compact Header ---
    with gr.Row():
        with gr.Column(scale=1, min_width=0):
            gr.HTML("""
                <div style="text-align: center; padding: 5px 0;">
                    <h1>📚 دستیار هوشمند تحلیل PDF</h1>
                    <p>سیستم پرسش و پاسخ هوشمند بر اساس اسناد شما.</p>
                    <div style="display: flex; justify-content: center; gap: 10px; font-size: 0.75em; color: #64748b; margin-top: 2px;">
                        <span><b>embeding model: </b> Tooka-SBERT 🧠</span>
                        <span> <b>LLM model: </b> Gemma-3-4B 🤖</span>
                    </div>
                </div>
            """)

    # --- Upload Section (Fixed 50px height) ---
    with gr.Row():
        with gr.Column(scale=4):
            pdf_input = gr.File(
                label="",  # Hide label to save space
                file_types=[".pdf"],
                height=50,  # Forced height
            )
        with gr.Column(scale=1, min_width=120):
            process_btn = gr.Button("⚙️ پردازش", variant="secondary")

    status_text = gr.Textbox(
        show_label=False,
        interactive=False,
        placeholder="منتظر آپلود فایل...",
        lines=1,
        max_lines=1,
    )

    retriever_state = gr.State()

    # --- Q&A Section ---
    with gr.Row():
        # Right Column (Questions)
        with gr.Column(scale=1):
            question_input = gr.Textbox(
                label="❓ سوال خود را بپرسید",
                placeholder="مثال: کدام گزینه سبب حذف خانه ی جاری...",
                lines=3,
            )
            ask_btn = gr.Button("🔍 دریافت پاسخ", variant="primary", interactive=False)

        # Left Column (Answers & References)
        with gr.Column(scale=1):
            answer_output = gr.Textbox(
                label="🤖 پاسخ هوش مصنوعی",
                lines=3,
                interactive=False,
                show_copy_button=True,
            )

            with gr.Accordion("📑 مشاهده متن استخراج شده (مرجع)", open=False):
                reference_output = gr.Textbox(
                    show_label=False, lines=2, interactive=False, elem_classes="ref-box"
                )

    # --- Event Listeners ---
    process_btn.click(
        fn=process_pdf, inputs=[pdf_input], outputs=[retriever_state, status_text, ask_btn]
    )

    ask_btn.click(
        fn=answer_question,
        inputs=[question_input, retriever_state],
        outputs=[answer_output, reference_output],
    )

    question_input.submit(
        fn=answer_question,
        inputs=[question_input, retriever_state],
        outputs=[answer_output, reference_output],
    )

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
