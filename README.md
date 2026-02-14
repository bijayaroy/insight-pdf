#  InsightPDF: AI-Powered Semantic Search

InsightPDF is a high-performance document intelligence application that enables users to perform semantic searches across multiple PDF documents. Unlike traditional keyword searching, InsightPDF utilizes Large Language Model (LLM) embeddings to understand the intent and context of your queries.

---

## 🌟 Key Features

- **Contextual Intelligence**  
  Powered by the `all-MiniLM-L6-v2` transformer model to find relevant information even when keywords don't match exactly.

- **Multi-Document Ingestion**  
  Seamlessly process and index multiple PDF files simultaneously.

- **Vectorized Search**  
  Leverages **Qdrant** for lightning-fast similarity searching within high-dimensional vector space.

- **Precise Metadata**  
  Every search result includes the source filename and the specific page number for easy verification.

- **Local & Secure**  
  Processes data locally using open-source models — no expensive API keys or external data sharing required.

---

## 🛠️ Tech Stack

- **Backend**: FastAPI (High-performance Python framework)  
- **Vector Engine**: Qdrant (In-memory configuration)  
- **ML Model**: Sentence-Transformers (HuggingFace)  
- **Document Parsing**: PyMuPDF (Engineered for speed and accuracy)  
- **Frontend**: Responsive UI built with Tailwind CSS  

---

## 📋 Prerequisites

- Python 3.8 or higher  
- pip package manager  

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone [https://github.com/YOUR_USERNAME/insight-pdf.git](https://github.com/YOUR_USERNAME/insight-pdf.git)
cd insight-pdf
```

---

### 2️⃣ Create a virtual environment

```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run the application

```bash
python main.py
```

The server will start at:

```text
http://127.0.0.1:8000
```

---

## 📖 How to Use

1. Launch the application in your browser.  
2. Use the **Upload** section to select your PDF documents.  
3. Once ingestion is complete, enter your query in the **Search bar**.  
4. Review the **top 5 most relevant excerpts**, complete with:
   - Relevance scores  
   - Source filename  
   - Page references  

---

## 📁 Project Structure

```text
InsightPDF/
│
├── main.py          # Core FastAPI logic, PDF processing, and Vector DB management
├── index.html       # Interactive web interface
├── requirements.txt # Project dependency manifest
├── .gitignore       # Configuration to keep the repository clean
```

---

## 📜 License

This project is open-source and available under the **MIT License**.

---

