# Local RAG-Powered Knowledge Assistant

## Project Description

This is a local full-stack app that answers questions about a small set of approved company documents. A React frontend sends a question to a Flask backend. The backend finds the most relevant passages in a Chroma vector database, sends them to a local Ollama model, and returns an answer with the sources it used.

Everything runs on one machine. Nothing is deployed.

## User Need and Scenario

Employees need quick answers about onboarding, product support, security practices, and workplace policies. Searching several documents by hand is slow, and a general chatbot can invent policies that do not exist. This assistant answers only from the approved documents and shows the passages it used, so an employee can check where an answer came from. That matters most for security and support procedures, where a wrong answer can cause real mistakes.

## Tech Stack

- Python 3.12, Flask, Flask-CORS
- Chroma (persistent local vector store)
- Ollama with `llama3.2` for answers and `nomic-embed-text` for embeddings
- React with Vite

## Project Structure

```text
client/                  React frontend (Vite)
server/
  app.py                 Flask app and /api/ask route
  config.py              Settings loaded from environment variables
  documents.py           Loads text files and splits them into chunks
  vector_store.py        Chroma client, embeddings, seeding, retrieval
  rag_service.py         RAG workflow: retrieve, build prompt, call model
  seed_knowledge_base.py Script that loads the knowledge base into Chroma
  knowledge_base/        onboarding.txt, product_support.txt,
                         security_guidelines.txt, workplace_faq.txt
  requirements.txt, Pipfile, .env.example
```

## Installation

You need Python 3.10 or newer, Node.js with npm, and Ollama installed and running.

1. Pull the models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

2. Set up the backend:

```bash
cd server
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

3. Set up the frontend:

```bash
cd client
npm install
```

## Environment Variables

Copy `server/.env.example` to `server/.env`. The defaults work for a standard local setup.

| Variable | Purpose | Default |
|---|---|---|
| `OLLAMA_BASE_URL` | URL of the local Ollama service | `http://localhost:11434` |
| `GENERATION_MODEL` | Model that writes answers | `llama3.2` |
| `EMBEDDING_MODEL` | Model that creates embeddings | `nomic-embed-text` |
| `CHROMA_PATH` | Folder where Chroma stores vector data | `./chroma_db` |
| `COLLECTION_NAME` | Name of the Chroma collection | `knowledge_assistant` |
| `KNOWLEDGE_BASE_PATH` | Folder with the source documents | `./knowledge_base` |
| `TOP_K` | Number of chunks to retrieve | `3` |
| `TEMPERATURE` | Controls answer variation | `0.2` |
| `CLIENT_ORIGIN` | Frontend origin allowed by CORS | `http://localhost:5173` |

## Running the App

Make sure the Ollama app is running.

1. Seed the vector database (run once, and again if the knowledge base files change):

```bash
cd server
source .venv/bin/activate
python seed_knowledge_base.py
```

You should see 4 documents loaded and 9 chunks stored. The generated `chroma_db` folder is not committed. Rebuild it with the command above.

2. Start the backend:

```bash
python app.py
```

3. In a second terminal, start the frontend:

```bash
cd client
npm run dev
```

4. Open `http://localhost:5173`.

## How to Verify the App

1. Open `http://localhost:5555/api/health`. It should return `{"message": "Backend is running.", "status": "ok"}`.
2. Open the frontend. The page should show "Backend is running."
3. Ask a question that the knowledge base covers. You should get an answer and a list of sources.
4. Ask a question the knowledge base does not cover. The assistant should say it does not have enough information.
5. Check the backend terminal and the browser console for errors.

## API Routes

### GET `/api/health`

Confirms the backend is running.

```json
{
  "status": "ok",
  "message": "Backend is running."
}
```

### POST `/api/ask`

Receives a question and returns an answer with sources.

Request body:

```json
{
  "question": "What should employees do with suspicious emails?"
}
```

Response:

```json
{
  "answer": "A helpful answer grounded in the knowledge base.",
  "sources": [
    {
      "title": "Security Guidelines",
      "source": "security_guidelines.txt",
      "chunk_index": 1,
      "excerpt": "Relevant source excerpt..."
    }
  ],
  "metadata": {
    "model": "llama3.2",
    "retrieved_chunks": 3
  }
}
```

If the question is missing or blank, the route returns a 400 error with a message. If the RAG workflow fails (for example, Ollama is not running), it returns a 500 error with a message.

## RAG Workflow

1. The seed script loads the four text files in `server/knowledge_base` and splits them into overlapping chunks of about 120 words.
2. Each chunk gets an embedding from `nomic-embed-text` through Ollama. The chunks are stored in a persistent Chroma collection with their source file, title, and chunk index.
3. When a question arrives at `/api/ask`, the backend creates an embedding for the question and retrieves the 3 closest chunks.
4. The backend builds a prompt that tells `llama3.2` to answer only from the retrieved context and to say so when the context is not enough.
5. The model's answer is returned to the frontend along with the retrieved chunks formatted as sources.
6. If no chunks are retrieved, the backend returns a "not enough information" message without calling the model.

## Design Notes

```text
User browser
→ React frontend (localhost:5173)
→ Flask POST /api/ask (localhost:5555)
→ rag_service.answer_question
→ vector_store.retrieve_relevant_chunks (embeds the question, queries Chroma)
→ Prompt built from the retrieved chunks
→ Ollama llama3.2 generates the answer
→ JSON with answer, sources, and metadata returned to the frontend
```

## Sample Questions Tested

| Sample Question | Was the Answer Relevant? | Were Useful Sources Returned? | Notes |
|---|---|---|---|
| What should employees do with suspicious emails? | Yes | Yes | Gave the three steps from the security guidelines: do not click links or download attachments, report through the security process, and disconnect and notify security if a link was clicked. Two of the three sources were Security Guidelines chunks. The third was an unrelated onboarding chunk. |
| What should a new employee complete during the first week? | Yes | Yes | Gave a numbered list of first-week tasks and cited the Employee Onboarding Guide. One of the three sources was an unrelated security chunk. |
| What is on the cafeteria menu today? | Yes | No | The knowledge base has nothing on this. The assistant said it did not have enough information and did not invent a menu. The three sources shown were unrelated chunks. |

## Known Limitations and Future Improvements

- Chroma always returns the closest chunks, even when none match well. Questions outside the knowledge base still show unrelated sources. A distance cutoff in `retrieve_relevant_chunks` would fix this, since the distance is already returned with each chunk.
- Chunks are split by word count, so some excerpts start or end in the middle of a sentence.
- The first request can be slow because `llama3.2` runs locally.
- The knowledge base is only four short documents, so retrieval quality on larger content is untested.
- Chroma prints a harmless "Failed to send telemetry event" message at startup.