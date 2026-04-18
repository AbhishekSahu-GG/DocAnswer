import os
import math
import gradio as gr
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# ── Config ───────────────────────────────────────────────────────────────
MODEL          = "openai/gpt-4o-mini"
EMBED_MODEL    = "text-embedding-3-large"
OPENROUTER_URL = "https://openrouter.ai/api/v1"
DB_NAME        = "/tmp/vector_db"          # /tmp is writable on HF Spaces
CHUNK_SIZE     = 1000
CHUNK_OVERLAP  = 300
RETRIEVE_K     = 5
TOP_K_FINAL    = 5
HISTORY_TURNS  = 5
# ─────────────────────────────────────────────────────────────────────────

load_dotenv(override=True)

# ── Per-session LLM & Embeddings factory ─────────────────────────────────
# We don't create llm/embeddings at module load time.
# Each user provides their own OpenRouter key via the UI.
# get_clients() builds fresh instances using that key.

def get_clients(api_key: str):
    """Returns (llm, embeddings) built with the user-supplied API key."""
    llm = ChatOpenAI(
        model=MODEL,
        temperature=0,
        base_url=OPENROUTER_URL,
        api_key=api_key,
    )
    embeddings = OpenAIEmbeddings(
        model=EMBED_MODEL,
        base_url=OPENROUTER_URL,
        api_key=api_key,
    )
    return llm, embeddings

# ── Prompts ───────────────────────────────────────────────────────────────
query_rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a search query optimizer. Rewrite the user's question into a \
clear, precise search query for a document retrieval system. You are also given the history of the chat which you can use.

Rules:
- Fix spelling and grammar
- Remove filler words like 'can you tell me', 'what does it say about'
- Make it specific and information-dense
- Keep it as ONE sentence
- Return ONLY the rewritten query, nothing else
- If input is a greeting or small talk, return it unchanged""",
    ),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

multi_query_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a query expansion assistant. Given a search query, generate \
exactly 3 different ways to ask the same question using different words. You are also given the history of the chat which you can use.

Return ONLY the 3 queries, one per line, no numbering, no extra text.

Example input: What are Python loops?
Example output:
How do loops work in Python programming?
Python iteration and loop structures explained
Types of loops available in Python""",
    ),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are DocAnswer, a friendly and helpful document assistant.

For greetings and small talk (hi, hello, thanks, how are you, etc.):
→ Respond warmly and naturally. Mention you're ready to help with their document.

For questions about document content:
→ Use ONLY the context provided below. Never use your own training knowledge.
→ Always mention the source file and page number when you answer.
→ If the answer is not in the context, say exactly: "I couldn't find this in the uploaded document."
→ Never generate code, examples, or facts that are not in the document.
→ Never guess or infer beyond what the document explicitly states.

Context from the document:
{context}""",
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

# ── Global state ──────────────────────────────────────────────────────────
# Both are None until the user provides a key and uploads a file.
retriever = None
_llm      = None


# ── Document Processing ───────────────────────────────────────────────────
def process_uploaded_file(api_key_input: str, file_paths):
    global retriever, _llm

    api_key_input = api_key_input.strip()
    if not api_key_input:
        return "⚠️ Please enter your OpenRouter API key first."

    if file_paths is None:
        return "⚠️ No file selected. Please upload a PDF."

    try:
        llm, embeddings = get_clients(api_key_input)
        _llm = llm          # save for use in answer_question

        all_chunks = []

        for file_path in file_paths:
            filename = os.path.basename(file_path)
            print(f"\n📄 Processing: {filename}")

            loader = PyMuPDFLoader(file_path)
            docs = loader.load()

            if not docs:
                return "❌ Could not extract text. PDF may be image-only or scanned."

            print(f"   Pages loaded: {len(docs)}")

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)
            print(f"   Chunks created: {len(chunks)}")

        # Rebuild vector DB fresh (stateless — no persistence needed)
        if os.path.exists(DB_NAME):
            Chroma(
                persist_directory=DB_NAME,
                embedding_function=embeddings,
            ).delete_collection()

        vectorstore = Chroma.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            persist_directory=DB_NAME,
        )
        print(f"   Vectors stored: {vectorstore._collection.count()}")

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RETRIEVE_K},
        )

        return f"✅ Ready! Processed {len(file_paths)} file(s) → {len(all_chunks)} chunks indexed.\nAsk your questions below."

    except Exception as e:
        return f"❌ Error: {str(e)}"


# ── Retrieval Functions ───────────────────────────────────────────────────
def rewrite_query(question: str, history, llm) -> str:
    chain = query_rewrite_prompt | llm | StrOutputParser()
    return chain.invoke({"question": question, "history": history}).strip()


def generate_multi_queries(question: str, history, llm) -> list:
    chain = multi_query_prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question, "history": history})
    queries = [q.strip() for q in result.strip().split("\n") if q.strip()]
    while len(queries) < 3:
        queries.append(question)
    return queries[:3]


def retrieve_all_chunks(queries: list) -> list:
    seen = set()
    all_docs = []
    for query in queries:
        docs = retriever.invoke(query)
        for doc in docs:
            fingerprint = doc.page_content[:100]
            if fingerprint not in seen:
                seen.add(fingerprint)
                all_docs.append(doc)
    return all_docs


def rerank_with_scores(question: str, docs: list, llm) -> list:
    scored = []
    score_prompt = (
        "Rate how relevant this text chunk is for answering the question below.\n"
        "Return ONLY a single integer from 0 to 10. Nothing else.\n\n"
        "Question: {question}\n\n"
        "Text chunk:\n{chunk}\n\n"
        "Score (0-10):"
    )
    for doc in docs:
        try:
            response = llm.invoke([HumanMessage(content=score_prompt.format(
                question=question,
                chunk=doc.page_content[:400],
            ))])
            digits = ''.join(filter(str.isdigit, response.content.strip().split()[0]))
            score = max(0, min(10, int(digits))) if digits else 0
        except Exception:
            score = 0
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def format_docs_with_sources(docs: list) -> str:
    if not docs:
        return "No relevant content found."
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        page   = doc.metadata.get("page", "?")
        formatted.append(f"[Source {i}: {source} | Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


# ── Evaluation Metrics ────────────────────────────────────────────────────
def compute_mrr(scores: list) -> float:
    for rank, score in enumerate(scores, 1):
        if score >= 5:
            return round(1.0 / rank, 3)
    return 0.0


def compute_recall_at_k(scores: list, k: int) -> float:
    total_relevant = sum(1 for s in scores if s >= 5)
    if total_relevant == 0:
        return 0.0
    found = sum(1 for s in scores[:k] if s >= 5)
    return round(found / total_relevant, 3)


def compute_ndcg(scores: list, k: int) -> float:
    def dcg(s):
        return sum(score / math.log2(i + 2) for i, score in enumerate(s[:k]))
    actual = dcg(scores)
    ideal  = dcg(sorted(scores, reverse=True))
    return round(actual / ideal, 3) if ideal > 0 else 0.0


def evaluate_retrieval(scored_docs: list) -> dict:
    scores = [s for s, _ in scored_docs]
    return {
        "MRR":           compute_mrr(scores),
        "Recall@5":      compute_recall_at_k(scores, k=5),
        "nDCG@5":        compute_ndcg(scores, k=5),
        "chunks_scored": len(scores),
        "avg_score":     round(sum(scores) / len(scores), 1) if scores else 0,
    }


# ── Small talk detection ──────────────────────────────────────────────────
GREETINGS = [
    "hi", "hello", "hey", "good morning", "good afternoon",
    "good evening", "thanks", "thank you", "bye", "goodbye",
    "how are you", "what's up", "whats up", "sup",
]


def is_small_talk(question: str) -> bool:
    q = question.lower().strip()
    return any(q.startswith(g) for g in GREETINGS)


# ── Main Chat Function ────────────────────────────────────────────────────
def answer_question(question: str, chat_history: list) -> tuple:
    if not question.strip():
        return "", chat_history

    if _llm is None:
        chat_history.append({"role": "user",      "content": question})
        chat_history.append({"role": "assistant", "content": "⚠️ Please enter your OpenRouter API key and process a PDF first."})
        return "", chat_history

    llm = _llm  # local alias

    try:
        history_messages = []
        for msg in chat_history[-(HISTORY_TURNS * 2):]:
            if msg["role"] == "user":
                history_messages.append(HumanMessage(content=msg["content"]))
            else:
                history_messages.append(AIMessage(content=msg["content"]))

        if is_small_talk(question) or retriever is None:
            if retriever is None and not is_small_talk(question):
                chat_history.append({"role": "user",      "content": question})
                chat_history.append({"role": "assistant", "content": "⚠️ Please upload a PDF first using the panel on the left."})
                return "", chat_history

            chain = answer_prompt | llm | StrOutputParser()
            answer = chain.invoke({
                "context":  "No document context needed — this is small talk.",
                "history":  history_messages,
                "question": question,
            })
            chat_history.append({"role": "user",      "content": question})
            chat_history.append({"role": "assistant", "content": answer})
            return "", chat_history

        # ── Full RAG Pipeline ─────────────────────────────────────────────
        rewritten  = rewrite_query(question, history_messages, llm)
        variations = generate_multi_queries(rewritten, history_messages, llm)
        all_queries = [rewritten] + variations

        raw_docs = retrieve_all_chunks(all_queries)

        if not raw_docs:
            chat_history.append({"role": "user",      "content": question})
            chat_history.append({"role": "assistant", "content": "I couldn't find relevant content in the uploaded document."})
            return "", chat_history

        scored_docs  = rerank_with_scores(rewritten, raw_docs, llm)
        eval_results = evaluate_retrieval(scored_docs)
        top_docs     = [doc for _, doc in scored_docs[:TOP_K_FINAL]]
        context      = format_docs_with_sources(top_docs)

        chain  = answer_prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "context":  context,
            "history":  history_messages,
            "question": question,
        })

        eval_note = (
            f"\n\n---\n"
            f"📊 *Retrieval quality — "
            f"MRR: {eval_results['MRR']} | "
            f"Recall@5: {eval_results['Recall@5']} | "
            f"nDCG@5: {eval_results['nDCG@5']} | "
            f"Chunks scored: {eval_results['chunks_scored']} | "
            f"Avg relevance: {eval_results['avg_score']}/10*"
        )

        chat_history.append({"role": "user",      "content": question})
        chat_history.append({"role": "assistant", "content": answer + eval_note})
        return "", chat_history

    except Exception as e:
        chat_history.append({"role": "user",      "content": question})
        chat_history.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})
        return "", chat_history


def clear_chat() -> list:
    return []


# ── Gradio UI ─────────────────────────────────────────────────────────────
with gr.Blocks(title="DocAnswer", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 📄 DocAnswer — Advanced RAG")
    gr.Markdown(
        "Upload a PDF → Process → Ask questions. "
        "Uses query rewriting, multi-query retrieval, and LLM reranking. "
        "Each answer includes retrieval quality scores."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🔑 Your OpenRouter API Key")
            api_key_box = gr.Textbox(
                placeholder="sk-or-v1-...",
                label="OpenRouter API Key",
                type="password",       # hides the key while typing
                lines=1,
            )
            gr.Markdown(
                "<small>Your key is used only for your session. "
                "Get a free key at [openrouter.ai](https://openrouter.ai/keys)</small>"
            )

            gr.Markdown("---")
            gr.Markdown("### 📁 Upload Document")
            file_upload = gr.File(
                label="Select PDFs",
                file_types=[".pdf"],
                type="filepath",
                file_count="multiple",
            )
            upload_btn = gr.Button("Process Document 🔄", variant="primary")
            upload_status = gr.Textbox(
                label="Status",
                value="Enter your API key, then upload a PDF.",
                interactive=False,
                lines=4,
            )

            gr.Markdown("---")
            gr.Markdown(
                "### 📊 Metric Guide\n\n"
                "**MRR** — Is the most relevant chunk ranked first?\n"
                "1.0 = perfect, 0.5 = it's rank 2\n\n"
                "**Recall@5** — Did we find all relevant chunks?\n"
                "1.0 = found everything relevant\n\n"
                "**nDCG@5** — Are best chunks ranked highest?\n"
                "1.0 = ideal ordering\n\n"
                "**Avg relevance** — Mean LLM score out of 10"
            )

        with gr.Column(scale=2):
            gr.Markdown("### 💬 Ask Questions")
            chatbot = gr.Chatbot(label="Conversation", height=520, type="messages")

            with gr.Row():
                question_box = gr.Textbox(
                    placeholder="Ask a question about your document...",
                    label="Your Question",
                    scale=8,
                )
                send_btn  = gr.Button("Send ➤",   variant="primary",   scale=1)
                clear_btn = gr.Button("Clear 🗑️", variant="secondary", scale=1)

    gr.Markdown(
        "**💡 Tips:** Say *hi* anytime for a normal greeting. "
        "Ask follow-ups naturally — the assistant remembers the last 5 turns. "
        "Upload a new PDF to switch documents."
    )

    # api_key_box is now passed as first input to process_uploaded_file
    upload_btn.click(fn=process_uploaded_file, inputs=[api_key_box, file_upload], outputs=[upload_status])
    send_btn.click(fn=answer_question, inputs=[question_box, chatbot], outputs=[question_box, chatbot])
    question_box.submit(fn=answer_question, inputs=[question_box, chatbot], outputs=[question_box, chatbot])
    clear_btn.click(fn=clear_chat, outputs=[chatbot])


# ── Launch ────────────────────────────────────────────────────────────────
# NOTE: No inbrowser=True — that flag tries to open a browser on the server
# which does nothing on HF Spaces. Removed intentionally.
demo.launch()
