# 🎬 Movie Recommendation System with OpenAI & Pinecone

This project demonstrates how to build a **semantic movie recommendation engine** using:

* **[Hugging Face Datasets](https://huggingface.co/datasets/AIatMongoDB/embedded_movies)** (movie metadata & plots)
* **OpenAI embeddings** (`text-embedding-3-small`) for semantic vector representations
* **[Pinecone](https://www.pinecone.io/)** as a vector database for scalable similarity search
* **OpenAI GPT model (`gpt-3.5-turbo`)** for conversational responses

---

## 🚀 Features

* Load and clean a movie dataset (remove missing plots, optimize embeddings).
* Generate embeddings for movie plots using OpenAI API.
* Store embeddings in **Pinecone** with hashed IDs for uniqueness.
* Perform **vector similarity search** for user queries.
* Use GPT to generate natural-language movie recommendations with context from search results.

---

###  Install Dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` should include:

```txt
datasets
pandas
openai
pinecone-client
python-dotenv
```

### Setup Environment Variables

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

---

## ▶️ Usage

### Run the script

```bash
python main.py
```

### Example Workflow

1. Load dataset (`AIatMongoDB/embedded_movies`).
2. Clean & generate embeddings for plots.
3. Store vectors in **Pinecone**.
4. Query system, e.g.:

   ```python
   query = "What is the best action movie to watch?"
   response, source_information = handle_user_query(query, index, dataset_df)
   print(response)
   ```
5. Output:

   ```
   Response: I recommend watching "Die Hard" — a classic action film...
   Source Information: 
   Title: Die Hard, Plot: John McClane takes on terrorists...
   ```

---

## 📊 Components

* **`get_embedding(text)`** → Generates OpenAI embeddings.
* **`batch_upsert(df)`** → Uploads embeddings into Pinecone in batches.
* **`vector_search(user_query, index)`** → Finds top similar movies by semantic meaning.
* **`handle_user_query(query)`** → Combines Pinecone results + GPT completion for recommendations.

---

## 🛡️ Notes & Best Practices

* Ensure **API keys** are kept private (use `.env`).
* Pinecone **index dimension** must match embedding size (`1536` for `text-embedding-3-small`).
* Avoid hitting **rate limits** on OpenAI API (consider batching).
* Extend system by adding:

  * More datasets
  * Fine-tuned embeddings
  * Genre filters

---

## 📌 Future Improvements

* Frontend UI (Streamlit or Flask) for interactive recommendations.
* Support for multiple embedding models.
* Personalized user profiling for recommendations.

---
