from flask import Flask, request, jsonify
from flask_cors import CORS

import os
import json
import requests

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://yellowsubmarineradar.netlify.app", "http://localhost:5500", "http://127.0.0.1:5050"]}})

# === CONFIG ===
API_KEY = "nk-LNuSNV-g7vHmrKmb_8t0CPCB6vVW8Hse0QkytrUYKYw"
DATASET_ID = "be957923-c51a-4118-aa7f-2224fd222284"
NOMIC_URL = "https://api-atlas.nomic.ai/v1/query/topk"

# === Vectorial ===

@app.route("/query", methods=["POST"])
def vector_query():
    """Realiza una consulta vectorial al dataset Atlas (endpoint oficial /v1/query/topk)."""
    data = request.get_json()
    user_query = data.get("query", "")
    model_filter = data.get("model", "")

    if not user_query:
        return jsonify({"error": "Missing 'query' field"}), 400

    body = {
        "projection_id": DATASET_ID,
        "k": 3,
        "fields": ["text", "metadata"],
        "query": user_query
    }

    if model_filter:
        body["selection"] = {
            "method": "composition",
            "conjunctor": "ALL",
            "filters": [
                {
                    "method": "search",
                    "query": model_filter,
                    "field": "metadata"
                }
            ]
        }

    try:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        response = requests.post(NOMIC_URL, headers=headers, data=json.dumps(body))

        if response.status_code != 200:
            return jsonify({
                "error": f"HTTP {response.status_code}",
                "details": response.text
            }), response.status_code

        return jsonify(response.json())

    except requests.exceptions.SSLError as e:
        return jsonify({"error": "SSL handshake failed", "details": str(e)}), 500
    except Exception as e:
        return jsonify({"error": "Request failed", "details": str(e)}), 500


# === RAG ===

def _extract_hits(response_json):
    """Extrae los textos y metadatos de los documentos relevantes."""
    hits = response_json.get("data") or response_json.get("hits") or []

    documents = []
    for hit in hits:
        text = hit.get("text", "")
        metadata_raw = hit.get("metadata", {})
        if isinstance(metadata_raw, str):
            try:
                metadata = json.loads(metadata_raw)
            except json.JSONDecodeError:
                metadata = {"raw": metadata_raw}
        else:
            metadata = metadata_raw

        documents.append({"text": text, "metadata": metadata})
    return documents


def _build_prompt(query, documents):
    """Construye el prompt para el LLM combinando la consulta y los documentos relevantes."""
    context = "\n\n".join([doc["text"] for doc in documents])
    prompt = (
        "Eres un asistente especializado en los manuales de Tesla. "
        "Responde de manera breve, clara y en español. "
        "Si la pregunta no tiene relación con los manuales Tesla, responde de forma amable "
        "indicando que solo puedes ayudar con temas del manual del vehículo.\n\n"
        f"Contexto relevante del manual:\n{context}\n\n"
        f"Pregunta del usuario: {query}\n"
        "Respuesta:"
    )
    return prompt


def _call_llm(prompt):
    """Llama al modelo Llama3.2 (servidor del curso) para obtener una respuesta al prompt."""
    llm_api_url = "https://asteroide.ing.uc.cl/api/generate"
    headers = {"Content-Type": "application/json"}
    body = {
        "model": "integracion",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.6,
            "num_ctx": 512
        }
    }

    response = requests.post(llm_api_url, headers=headers, json=body, timeout=120)
    response.raise_for_status()
    data = response.json()

    return data.get("response") or data.get("output_text", "")


@app.route("/rag_query", methods=["POST"])
def rag_query():
    """Realiza una consulta RAG (Retrieval-Augmented Generation)."""
    data = request.get_json()
    user_query = data.get("query", "")
    model_filter = data.get("model", "")

    if not user_query:
        return jsonify({"error": "Missing 'query' field"}), 400

    body = {
        "projection_id": DATASET_ID,
        "k": 3,
        "fields": ["text", "metadata"],
        "query": user_query
    }

    if model_filter:
        body["selection"] = {
            "method": "composition",
            "conjunctor": "ALL",
            "filters": [
                {
                    "method": "search",
                    "query": model_filter,
                    "field": "metadata"
                }
            ]
        }

    try:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        response = requests.post(NOMIC_URL, headers=headers, data=json.dumps(body))
        if response.status_code != 200:
            return jsonify({
                "error": f"HTTP {response.status_code}",
                "details": response.text
            }), response.status_code

        response_json = response.json()
        documents = _extract_hits(response_json)
        prompt = _build_prompt(user_query, documents)
        llm_response = _call_llm(prompt)

        return jsonify({
            "query": user_query,
            "documents": documents,
            "response": llm_response
        })

    except requests.exceptions.RequestException as e:
        return jsonify({"error": "Request failed", "details": str(e)}), 500
    except Exception as e:
        return jsonify({"error": "Internal error", "details": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
