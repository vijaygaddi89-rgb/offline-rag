# offline-rag
## Tech Stack
| Tool | Purpose |
|------|---------|
| Llama 3.2 3B Q4_K_M | Local LLM inference via llama.cpp |
| ChromaDB | Local vector database |
| all-MiniLM-L6-v2 | Local embedding model |
| PyMuPDF | PDF text extraction |
| Fernet AES | File encryption |
| Streamlit | Chat UI |

## How to Run
```bash
git clone https://github.com/vijaygaddi89-rgb/offline-rag
cd offline-rag

py -m venv rag
rag\Scripts\activate
pip install -r requirements.txt

# Download Llama-3.2-3B-Instruct-Q4_K_M.gguf from HuggingFace
# Place it inside the models/ folder

streamlit run app.py
```
![Offline Demo](screenshots/demo.png)

## Key Features
- ✅ Zero internet required after setup
- ✅ All data stays on your device
- ✅ Original files encrypted with AES
- ✅ Page-level citations on every answer
- ✅ Full audit log of every query
- ✅ Works on standard laptop hardware 16GB RAM
- ✅ Running cost ₹0 per query

---
Built by Vijay Gaddi · B.Tech CSE Data Science · MGIT Hyderabad · 2025