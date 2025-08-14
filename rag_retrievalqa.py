from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
try:
    from torch.ao.quantization import quantize_dynamic as qd
except Exception:
    from torch.quantization import quantize_dynamic as qd
import torch


emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)
vs = FAISS.load_local("faiss_index", emb, allow_dangerous_deserialization=True)
retriever = vs.as_retriever(search_kwargs={"k": 2})


tok = AutoTokenizer.from_pretrained("google/flan-t5-small", use_fast=True)
mdl = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
mdl = qd(mdl, {torch.nn.Linear}, dtype=torch.qint8)
mdl.eval()
gen = pipeline("text2text-generation", model=mdl, tokenizer=tok, device=-1, max_new_tokens=120)
llm = HuggingFacePipeline(pipeline=gen)


qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",                
    return_source_documents=True
)


res = qa.invoke({"query": "What is machine learning?"})
print(res["result"])
for i, d in enumerate(res["source_documents"], 1):
    print(f"[{i}] {d.metadata.get('source')}")
