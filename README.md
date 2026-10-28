# Experiment 7 — Customer Support RAG API

FastAPI REST API for the Experiment 7 lab. It exposes:
- `GET /health` — API process health
- `POST /ask` — ask a question against the included fictional FAQ
- `/docs` — interactive Swagger UI

## Important scope note
The supplied notebook's default backend is `mock` (deterministic, offline, no API key). This is ideal for demonstrating the API and endpoint tests, but it is **not a generative LLM**. The notebook also includes optional Ollama and Hugging Face backends. The bundled FAQ is a lightweight placeholder based on the customer-support domain described in the notebook; replace it with your approved Experiment 5/6 assets if those are required.

## Run locally (Ubuntu/Linux/macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs`, then try `POST /ask` with:
```json
{"question": "How do I reset my password?"}
```

Or test from another terminal:
```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"How do I reset my password?"}'
```

## Deploy the API publicly on Render (simple demo deployment)

1. Create a GitHub repository and upload these project files (`app.py`, `support_rag.py`, `requirements.txt`, `.env.example`, `.gitignore`, `README.md`, `render.yaml`). Do **not** upload `.env`, access tokens, or private data.
2. Sign in to Render and choose **New → Web Service**, then connect the repository.
3. Use these settings:
   - Runtime: Python
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
4. Add environment variable `RAG_BACKEND=mock` for the first deployment. Deploy.
5. Once Render shows the service as Live, visit `https://YOUR-SERVICE.onrender.com/health` and `https://YOUR-SERVICE.onrender.com/docs`.
6. In `/docs`, expand `POST /ask` → **Try it out** → enter JSON → **Execute**.

A `render.yaml` Blueprint is included as an alternative: in Render, create a new Blueprint and select the repository.

### Deploying with an actual model
- **Ollama:** this project calls an Ollama server at `OLLAMA_BASE_URL` (default `http://localhost:11434`). A normal Render web service does not automatically run Ollama or include model weights. Do not select `RAG_BACKEND=ollama` on Render unless you have separately deployed a reachable Ollama service and configured `OLLAMA_BASE_URL` and `OLLAMA_MODEL`.
- **Hugging Face:** set `RAG_BACKEND=hf`, `HF_TOKEN`, and `HF_MODEL` in the hosting provider's environment-variable settings. Never put the token in source code or commit `.env`. Provider/model access and quotas vary. Test this backend before relying on it.
- For a straightforward lab demo, deploy with `mock` first, then configure a real model only if required.

## API behavior
- `POST /ask` accepts a JSON body with a non-empty `question` up to 1,000 characters.
- Invalid/blank/overlong input returns HTTP 422.
- Backend errors return HTTP 503 with a generic message.
- Response includes answer, sources, backend and latency in milliseconds.
- Logs avoid recording the full question.

## Limitations / production hardening
This is a lab baseline, not a production-ready public service. Before real public use, add authentication, rate limiting, HTTPS policy verification, monitoring, abuse controls, stronger retrieval/evaluation, and an operational recovery plan. Do not use real customer secrets or sensitive data in this demo.
