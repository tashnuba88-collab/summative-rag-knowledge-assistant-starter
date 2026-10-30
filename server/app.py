from flask import Flask, jsonify, request
from flask_cors import CORS

from config import Config
from rag_service import answer_question

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": Config.CLIENT_ORIGIN}})


@app.get("/api/health")
def health():
    """Confirm that the backend is running."""
    return jsonify(
        {
            "status": "ok",
            "message": "Backend is running.",
        }
    ), 200


@app.post("/api/ask")
def ask_question():
    """Receive a question from the frontend and return an answer with sources."""
    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()

    if not question:
        return jsonify({"error": "Question is required."}), 400

    try:
        result = answer_question(question)
        return jsonify(result), 200
    except Exception as exc:
        app.logger.exception("RAG workflow failed")
        return jsonify(
            {
                "error": f"Something went wrong while answering: {exc}",
                "sources": [],
            }
        ), 500


if __name__ == "__main__":
    app.run(debug=True, port=5555)