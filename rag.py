import os
import argparse
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import gradio as gr
import shutil

try:
    from torch.ao.quantization import quantize_dynamic as qd
except Exception:
    from torch.quantization import quantize_dynamic as qd

def get_embeddings(emb_model: str):
    return HuggingFaceEmbeddings(
        model_name=emb_model,
        encode_kwargs={"normalize_embeddings": True}
    )

def load_docs(docs_dir: str):
    docs_dir_path = Path(docs_dir)
    all_docs = []
    for p in docs_dir_path.glob("*.txt"):
        all_docs.extend(TextLoader(str(p), encoding="utf-8").load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)
    return chunks

def build_index_from_docs(docs_dir: str, embeddings):
    chunks = load_docs(docs_dir)
    if not chunks:
        raise SystemExit(f"No .txt files found in '{docs_dir}'")
    return FAISS.from_documents(chunks, embeddings)

def load_index(index_dir: str, embeddings):
    return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)

def answer_with_context(context_text: str, question: str, llm_name: str):
    tok = AutoTokenizer.from_pretrained(llm_name, use_fast=True)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(llm_name)
    mdl = qd(mdl, {torch.nn.Linear}, dtype=torch.qint8)
    mdl.eval()
   
    prompt = (
        "Answer the question using only the provided context. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\nAnswer:"
    )

    inputs = tok(prompt, return_tensors="pt", truncation=True)
    gen_kwargs = dict(
        max_new_tokens=200,
        do_sample=False,         
        num_beams=1,             
        eos_token_id=tok.eos_token_id,
    )
    with torch.no_grad():
        output = mdl.generate(**inputs, **gen_kwargs)

    answer = tok.decode(output[0], skip_special_tokens=True)
    if "Answer:" in answer:
        answer = answer.split("Answer:", 1)[-1].strip()

    return answer

def load_or_build_index(index_dir: str, docs_dir: str, emb_model: str):
    embeddings = get_embeddings(emb_model)
    if os.path.exists(index_dir):
        print("Loading existing FAISS index...")
        vs = load_index(index_dir, embeddings)
    else:
        print("Building new FAISS index...")
        vs = build_index_from_docs(docs_dir, embeddings)
        vs.save_local(index_dir)
    return vs, embeddings

def run_cli(args):
    vectorstore, _ = load_or_build_index(args.index_dir, args.docs_dir, args.emb_model)

    hits = vectorstore.similarity_search(args.query, k=args.k)
    context = "\n\n---\n\n".join(d.page_content for d in hits)

    answer = answer_with_context(context, args.query, args.llm)

    print("\n=== Retrieved context (top-k) ===")
    for i, d in enumerate(hits, 1):
        src = d.metadata.get("source", "unknown")
        preview = d.page_content[:200].replace("\n", " ")
        print(f"[{i}] ({src}) {preview}...\n")

    print("=== Answer ===")
    print(answer)

def run_gradio(args):
    
    vectorstore, embeddings = load_or_build_index(args.index_dir, args.docs_dir, args.emb_model)
    tok = AutoTokenizer.from_pretrained(args.llm, use_fast=True)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(args.llm)
    mdl.eval()
    state = {"vs": vectorstore, "emb": embeddings}

    def ask_ui(query, k):
        if not query or not query.strip():
            return "Please enter a question.", ""
        hits = state['vs'].similarity_search(query, k=int(k))
        if not hits:
            return "No context found.", ""
        context = "\n\n---\n\n".join(d.page_content for d in hits)

        
        prompt = (
            "Answer the question using only the provided context. "
            "If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\nAnswer:"
        )
        inputs = tok(prompt, return_tensors="pt", truncation=True)
        with torch.no_grad():
            out = mdl.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                num_beams=1,
                eos_token_id=tok.eos_token_id,
            )
        ans = tok.decode(out[0], skip_special_tokens=True)
        if "Answer:" in ans:
            ans = ans.split("Answer:", 1)[-1].strip()
        
        lines = []
        for i, d in enumerate(hits, 1):
            src = d.metadata.get("source", "unknown")
            preview = d.page_content[:180].replace("\n", " ")
            lines.append(f"[{i}] ({src}) {preview}...")
        sources = "\n\n".join(lines)
        return ans, sources
    
    def rebuild_index_from_files(uploaded_files):
        
        docs_dir = Path(args.docs_dir)
        docs_dir.mkdir(parents=True, exist_ok=True)
        for f in uploaded_files or []:
            dst = docs_dir / Path(f.name).name
            shutil.copy(f.name, dst)

        
        new_vs = build_index_from_docs(str(docs_dir), state["emb"])
        new_vs.save_local(args.index_dir)

        state["vs"] = new_vs
        return "Index rebuilt successfully. You can ask questions now."


    with gr.Blocks(title="Local RAG (CPU)") as demo:
        gr.Markdown("# Local RAG (CPU)\nAsk questions about your local `.txt` docs.")
        with gr.Row():
            q = gr.Textbox(label="Question", placeholder="What is machine learning?")
            k = gr.Slider(1, 5, value=3, step=1, label="Top-k chunks")
        ans = gr.Textbox(label="Answer", lines=8)
        src = gr.Textbox(label="Retrieved context (sources)", lines=10)
        btn = gr.Button("Ask")
        btn.click(ask_ui, inputs=[q, k], outputs=[ans, src])
        files = gr.File(label="Upload .txt files", file_count="multiple", file_types=[".txt"])
        reindex_btn = gr.Button("Reindex")
        status = gr.Textbox(label="Status", interactive=False)
        reindex_btn.click(rebuild_index_from_files, inputs=files, outputs=status)
    demo.launch()
    
    
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs_dir", default="docs", help="Folder with .txt files")
    ap.add_argument("--index_dir", default="faiss_index", help="Folder to save/load FAISS index")
    ap.add_argument("--query", required=True, help="User question")
    ap.add_argument("--k", type=int, default=3, help="Top-k chunks to retrieve")
    ap.add_argument("--emb_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--llm", default="google/flan-t5-small")
    ap.add_argument("--gradio", action="store_true", help="Launch Gradio UI instead of CLI")
    args = ap.parse_args()

    if args.gradio:
        run_gradio(args)
    else:
        if not args.query:
            raise SystemExit("Please provide --query in CLI mode, or run with --gradio.")
        run_cli(args)

    
if __name__ == "__main__":
    main()
