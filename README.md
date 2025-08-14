# Local RAG (CPU) — Text Search + LLM Answering

This project demonstrates a simple **Retrieval-Augmented Generation (RAG)** pipeline running entirely on CPU.

[![Open in Spaces](https://img.shields.io/badge/🤗%20Open%20in%20Spaces-blue?logo=huggingface)](https://huggingface.co/spaces/stefaniabenea/rag-llm-gradio)

Features:
- Load `.txt` documents from local folder
- Build a **FAISS vector store** with text embeddings
- Search top-k relevant chunks for a query
- Use a **quantized** LLM (INT8) for answering
- Run via CLI or interactive **Gradio UI**
- Upload `.txt` files and reindex from the browser

## 1. Installation
```
pip install -r requirements.txt
```
## 2. Usage
### CLI mode
```
python app.py --query "What is machine learning?"
```

### Gradio UI mode
```
python app.py --query "What is machine learning?" --gradio
```
## 3. Options
```
| Parameter    | Description                          | Default                                |
| ------------ | ------------------------------------ | -------------------------------------- |
| --docs\_dir  | Folder containing `.txt` documents   | docs                                   |
| --index\_dir | Folder to save/load FAISS index      | faiss\_index                           |
| --query      | User question (required in CLI mode) | -                                      |
| --k          | Top-k chunks to retrieve             | 3                                      |
| --emb\_model | HuggingFace embedding model          | sentence-transformers/all-MiniLM-L6-v2 |
| --llm        | HuggingFace LLM for answering        | google/flan-t5-small                   |
| --gradio     | Launch Gradio UI instead of CLI mode | disabled                               |
```
## 4. Example .txt file
docs/machine_learning.txt
Machine learning is a subfield of artificial intelligence focused on building systems that learn from data...

## 5. Example CLI run
```
python app.py --query "What is deep learning?"
```
Output:
```
=== Retrieved context (top-k) ===
[1] (docs\deep_learning.txt) Deep learning is a subset of machine learning that uses neural networks with many layers. These models are particularly good at processing images, audio, and natural language. Convolutional neural net...

[2] (docs\cnn_fundamentals.txt) Convolutional Neural Networks (CNNs) specialize in grid-like data such as images. Key components are convolution layers, non-linear activations, pooling, and fully connected layers. Data augmentation ...

[3] (docs\machine_learning.txt) Machine learning is a field of artificial intelligence that focuses on enabling computers to learn from data without being explicitly programmed. It includes supervised, unsupervised, and reinforcemen...

=== Answer ===
a subset of machine learning that uses neural networks with many layers.
```
## 6. Gradio UI

The UI allows:
- Asking questions in a textbox
- Adjusting top-k chunks
- Viewing retrieved context
- Uploading .txt files and reindexing without restarting

## 7. Notes
- Runs entirely on CPU (tested with quantized models for speed)
- FAISS index persists in faiss_index/
- Embeddings are computed once unless you reindex

## 8. Alternative: Minimal RetrievalQA Script
If you prefer a shorter implementation, you can use LangChain's `RetrievalQA` class.  
It loads the FAISS index, retrieves the context, and calls the model directly without manual prompt construction.  
See the `rag_retrievalqa.py` file in this repo.
