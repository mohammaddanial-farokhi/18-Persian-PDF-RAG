# Persian PDF RAG Assistant (Gemma-3-4B + Tooka-SBERT)

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![LangChain](https://img.shields.io/badge/LangChain-Framework-green.svg)
![Unsloth](https://img.shields.io/badge/Unsloth-Inference-orange.svg)
![Gradio](https://img.shields.io/badge/UI-Gradio-ff69b4.svg)
![Task](https://img.shields.io/badge/Task-RAG%20(Retrieval--Augmented%20Generation)-blueviolet.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

---

## 🧠 About The Project

A complete end-to-end **Retrieval-Augmented Generation (RAG)** system designed specifically for Persian (Farsi) PDF documents. 

This project allows users to upload a PDF file, extract its text, and ask questions based on the document's content. It utilizes **Unsloth** for extremely fast, 4-bit quantized inference of the **Gemma-3-4B-IT** model, and **Tooka-SBERT** for generating high-quality Persian semantic embeddings. The system is served via a beautifully designed, fully responsive **Gradio** web interface with strict RTL (Right-to-Left) support, optimized for Hugging Face Spaces (ZeroGPU).

**Result:** The system accurately retrieves context from Persian PDFs, handles mixed English/Persian technical terms gracefully, and generates conversational, human-like responses without hallucinations.

---

## 🎯 Key Features

1. **Efficient Inference:** Uses Unsloth's `FastLanguageModel` for blazing-fast 4-bit quantized inference of Gemma-3.
2. **Persian-Optimized Embedding:** Utilizes `PartAI/Tooka-SBERT-V2-Large` to ensure accurate semantic search within Persian text.
3. **Smart Chunking & Retrieval:** Employs LangChain's `RecursiveCharacterTextSplitter` (chunk size: 600, overlap: 100) and retrieves the top 3 (`k=3`) most relevant chunks.
4. **Conversational Prompt Engineering:** Custom-engineered prompts ensure the model acts as a polite tutor, avoids robotic responses, prevents "hallucinations" (making things up), and suppresses repetitive closing phrases (e.g., "Hope this helps!").
5. **Premium Web UI:** A custom Gradio interface featuring modern glassmorphism CSS, Vazirmatn font, single-screen layout, and side-by-side input/output panels with a collapsible reference accordion.
6. **ZeroGPU Ready:** Fully prepared for Hugging Face Spaces using the `@spaces.GPU` decorator and custom environment handling.

---

## 📂 Project Structure

```text
.
├── main.py               # Core CLI pipeline (Model loading, PDF processing, RAG logic)
├── app.py                # Gradio web UI application (For local testing & HF Spaces)
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## ⚙️ System Architecture & Configuration

| Parameter | Value | Note |
| :--- | :--- | :--- |
| **LLM Base** | `unsloth/gemma-3-4b-it-unsloth-bnb-4bit` | Pre-quantized 4-bit model |
| **Embedding Model** | `PartAI/Tooka-SBERT-V2-Large` | Persian-specific Sentence Transformer |
| **Vector Store** | `DocArrayInMemorySearch` | Fast, in-memory semantic search |
| **Chunk Size** | 600 characters | Optimized for dense Persian Q&A |
| **Chunk Overlap** | 100 characters | Preserves context across boundaries |
| **Retrieval `k`** | 3 | Top 3 chunks fed into the LLM |
| **Max New Tokens** | 512 | Prevents infinite generation |
| **Temperature** | 0.1 | Ensures factual, deterministic answers |

---

## 🚀 Usage

### 1. Install Dependencies
Ensure you have Python 3.10+ and CUDA installed. Then install the required packages:
```bash
pip install -r requirements.txt
```

### 2. CLI Version (Testing & Debugging)
To test the RAG pipeline directly in your terminal, run:
```bash
python main.py
```
*(Note: Ensure you have a `test_sample.pdf` file in the root directory before running).*

### 3. Web Application (Local & Hugging Face Spaces)
To launch the beautiful Gradio UI, run:
```bash
python app.py
```
Open `http://127.0.0.1:7860` in your browser. The interface allows you to drag and drop your PDF, ask questions, and view the exact source text used to generate the answer.

### 4. Deploying to Hugging Face Spaces
This project is ready to be deployed to Hugging Face Spaces using **ZeroGPU**:
1. Create a new Space with the **Gradio** SDK.
2. Select **ZeroGPU** as the hardware.
3. Upload `app.py` and `requirements.txt`.
4. Ensure the `README.md` contains the necessary YAML frontmatter (`sdk: gradio`, `app_file: app.py`).

---

## ⚠️ Known Limitations

*   **PDF Text Extraction:** Persian PDFs often suffer from RTL/LTR formatting issues (e.g., numbers extracting as `27/1` instead of `1.27`). The system attempts to handle these, but extreme formatting anomalies may confuse the model.
*   **ZeroGPU Environment:** Running `unsloth` and `torch` inside Hugging Face Spaces requires specific import ordering and environment variables (`UNSLOTH_ALLOW_CPU=1`) to prevent CUDA initialization crashes before GPU allocation.
*   **Context Window:** Extremely long multi-part questions combined with 3 retrieved chunks might hit the `max_seq_length=2048` limit. 

---

## 📜 License
This project is licensed under the MIT License - free to use, modify, and distribute.
