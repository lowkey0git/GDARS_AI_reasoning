# GDARS v2 — Goal-Directed Adaptive Reasoning System

An autonomous AI reasoning system built on LangGraph with a hierarchical
planner, mixture-of-experts model routing, three-layer memory, and a
lightweight evaluation harness that scores reasoning steps in real time.

---

## What makes this different from v1

| Concern | v1 | v2 |
|---|---|---|
| Model routing | One 70B model for everything | MoE: 8B for routing/scoring, 70B for reasoning, 8B-instant for synthesis |
| Query routing | Flat loop, all nodes every time | Hierarchical: SIMPLE (3 nodes) / COMPOUND (6) / NOVEL (full) |
| Memory | ChromaDB + Pinecone | 3-layer: Redis working / ChromaDB episodic / FAISS semantic |
| Reasoning evaluation | None | Step-level factuality + usefulness scores mid-chain |
| Personalization | 5 sequential LLM calls | 1 call with injected style instructions |
| Node count | 9 | 8 (synthesis merged personalization + final) |
| Response UX | One long wait | Progressive SSE stream (goal → evidence → reasoning → answer) |

---

## Architecture

```
Query in
  │
  ▼
planner_node ─── classifies complexity (8B router)
                 infers true goal (70B, skipped for SIMPLE)
                 builds execution plan (70B, skipped for SIMPLE)
  │
  ▼
memory_node ──── reads all 3 memory layers
  │
  ├─ SIMPLE ──────────────────────────────────────────┐
  ▼                                                   │
retrieval_node ─ query decomposition                  │
                 parallel: Tavily + FAISS + ChromaDB  │
                 scoring → dedup → synthesis          │
  │                                                   │
  ├─ NOVEL with tool steps ─────┐                     │
  ▼                             ▼                     │
execution_node              reasoning_node ◄──────────┘
(web search, codegen)          (70B CoT)
                               + mid-chain eval (8B)
                               + step retry on low score
  │
  ▼
synthesis_node ─ style adaptation + final assembly (8B-instant)
  │
  ▼
verification_node ─ hallucination check + confidence (8B)
  │
  ├─ NOVEL + low confidence ──→ reflection_node ──→ reasoning_node
  │
  └─ done ──→ END

Answer out (streaming or complete)
```

---

## Mixture of Experts Routing

| Model | Role | Why |
|---|---|---|
| `llama3-8b-8192` | Query classification, step scoring, verification | Fast, cheap, deterministic — doesn't need 70B capacity |
| `llama-3.3-70b-versatile` | Goal inference, planning, deep CoT | Full capacity needed for complex reasoning |
| `llama-3.1-8b-instant` | Synthesis / final response | Formatting an existing answer doesn't need 70B |
| `llama-3.3-70b-versatile` | Code generation | Same as reasoning — correctness matters |

All models run on Groq (free tier available).

---

## Three-Layer Memory

| Layer | Backend | Scope | Use |
|---|---|---|---|
| Working | Redis | Session (1hr TTL) | Recent turns, in-flight context |
| Episodic | ChromaDB | Persistent | Past Q&A, inferred goals |
| Semantic | FAISS | Persistent | Your domain knowledge base |

The planner escalates through layers: working → episodic → semantic.
Simple queries never touch episodic or semantic, saving ~35ms and 2 embedding calls.

---

## Evaluation Harness

Every `ThoughtStep` gets scored on:
- **Factuality** (0–1): is the claim grounded in retrieved evidence?
- **Usefulness** (0–1): does this step move reasoning toward the goal?

For NOVEL queries, scoring happens after each step. A step below threshold
is retried once with a corrective prompt before the chain continues. This
catches bad reasoning early instead of propagating it through 6 more steps.

---

## Project Structure

```
gdars2/
├── agents/
│   ├── state.py          # AgentState with ThoughtStep, QueryComplexity, StepScore
│   ├── graph.py          # LangGraph StateGraph with hierarchical routing
│   ├── planner.py        # MoE router + goal inference + plan generation
│   ├── retrieval.py      # Parallel multi-source retrieval + scoring + dedup
│   ├── reasoning.py      # CoT with mid-chain evaluation and step retry
│   ├── synthesis.py      # Style adaptation + final assembly (1 LLM call)
│   └── verification.py   # Hallucination check + reflection trigger
├── evaluation/
│   └── harness.py        # Step-level factuality and usefulness scoring
├── memory/
│   └── memory_system.py  # Redis + ChromaDB + FAISS unified facade
├── tools/
│   └── registry.py       # Tavily search + code generation
├── api/
│   └── app.py            # FastAPI: /health /ready /query /query/stream /metrics
├── config/
│   └── settings.py       # Grouped Pydantic settings with env prefix per module
├── utils/
│   ├── logging.py        # structlog on stdlib backend (no PrintLogger crash)
│   └── llm_client.py     # GroqClient + MoE factory functions
├── tests/
│   ├── conftest.py       # autouse logging + settings mocks
│   ├── test_state.py     # Pydantic validation, helpers, routing logic
│   ├── test_planner.py   # Complexity routing + goal inference + fallbacks
│   ├── test_evaluation.py# Step scoring, thresholds, state aggregation
│   ├── test_memory.py    # All 3 memory layers
│   └── test_api.py       # All endpoints, error cases, field contracts
├── .github/workflows/
│   └── cicd.yml          # test → build → ECS deploy
├── main.py
├── Dockerfile
├── docker-compose.yml    # API + Redis
├── requirements.txt
├── pytest.ini
└── .env.example
```

---

## Quick Start

### Local (no Docker)

```powershell
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create directories
mkdir data\chroma
mkdir data\faiss_index
mkdir logs

# 4. Configure
copy .env.example .env
notepad .env
# Add: GROQ_API_KEY and RETRIEVAL_TAVILY_API_KEY

# 5. Start Redis (Docker)
docker run -d -p 6379:6379 redis:7-alpine

# 6. Run tests
set PYTHONPATH=.
pytest tests/ -v --tb=short

# 7. Start server
python main.py
# → http://localhost:8000/docs
```

### Docker Compose (recommended)

```bash
cp .env.example .env
# fill in GROQ_API_KEY and RETRIEVAL_TAVILY_API_KEY

docker compose up --build
# API at http://localhost:8000
```

---

## API

### `POST /query`

```json
{
  "query": "How do I build a production REST API in Python?",
  "user_profile": {
    "skill_level": "intermediate",
    "code_oriented": true,
    "tone": "direct"
  }
}
```

**Response includes:**
- `final_solution` — the answer
- `complexity` — simple / compound / novel
- `confidence_score` — 0–1
- `avg_factuality` / `avg_usefulness` — reasoning quality scores
- `models_used` — MoE audit trail
- `node_trace` — which nodes ran
- `execution_time_ms`

### `POST /query/stream` — SSE streaming

Emits events as each node completes. Client can render progressively:
1. `planner_node` → shows inferred goal + complexity
2. `retrieval_node` → shows evidence summary
3. `reasoning_node` → shows each conclusion as it's produced
4. `synthesis_node` → shows partial answer
5. `verification_node` → shows confidence
6. `__end__` → complete

### `GET /health` — always 200, shows per-component status
### `GET /ready` — 503 until initialised, used by ECS/K8s
### `POST /memory/ingest` — add documents to semantic memory
### `GET /metrics` — request and error counts

---

## Deploy to AWS

```bash
# Prerequisites: AWS CLI + Terraform installed

# 1. Bootstrap remote state
./scripts/bootstrap_aws.sh

# 2. Configure
cd terraform
cp terraform.tfvars.example terraform.tfvars
# fill in groq_api_key, tavily_api_key

# 3. Provision
terraform init
terraform apply -var-file=terraform.tfvars

# 4. Add GitHub Actions secrets
terraform output cicd_access_key_id
terraform output -raw cicd_secret_access_key

# 5. Push to main → automatic deploy
```

---

## CI/CD Pipeline

```
push to any branch  → tests run (pytest + coverage)
push to any branch  → build + push Docker image to ECR
push to main only   → deploy to ECS Fargate
                    → smoke test /ready + /health
                    → auto-rollback if deployment fails
```

GitHub Actions secrets needed: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
