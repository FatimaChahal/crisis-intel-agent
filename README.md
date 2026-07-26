# 🚨 Crisis Intel Agent

> **Plateforme agentique de veille et d'analyse de crises géospatiales**  
> Multi-agent · Agentic RAG · MLOps · Cloud AWS · API REST · CI/CD

---

## 🎯 Objectif

Crisis Intel Agent automatise la **veille, l'analyse et le résumé de situations de crise** (inondations, risques environnementaux, événements géospatiaux) à partir de sources de données hétérogènes (news, données GIS, capteurs).

Le système est construit autour d'une architecture **multi-agents orchestrée avec LangGraph**, d'un pipeline **Agentic RAG évalué**, et d'un déploiement **MLOps-grade sur AWS ECS Fargate**.

---

## 🏗️ Architecture

```
Sources (News / GIS / Capteurs)
        │
        ▼
┌──────────────────────────────────────────┐
│    Data Pipeline — Medallion (S3)        │
│  Bronze (brut) → Silver (propre) → Gold  │
│  Pandas · Pydantic · JSON · boto3        │
└──────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│       Scout Agent (LangGraph)            │
│                                          │
│   Retrieve ──▶ Analyse ──▶ Rapport       │
│   ChromaDB     Groq/Llama   Structuré    │
│   (RAG)        3.3 70B                   │
└──────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│        FastAPI REST API                  │
│   GET /health  POST /ingest              │
│   Sécurisée par API Key (X-API-Key)      │
└──────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│     Observabilité & MLOps                │
│  Langfuse (tracing LLM) · MLflow         │
│  GitHub Actions CI/CD (black + pytest)   │
└──────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│     Cloud AWS                            │
│  Docker → ECR → ECS Fargate              │
│  S3 · CloudWatch · Security Groups       │
└──────────────────────────────────────────┘
```

---

## 🔄 Exemple concret — alerte inondation

**Entrée (Bronze — brute) :**
```json
{
    "titre": "  FLOOD IN GERMANY  ",
    "pays": "  germany  ",
    "severite": "  ORANGE  "
}
```

**Silver — nettoyée (Pandas + Pydantic) :**
```json
{
    "titre": "flood in germany",
    "pays": "germany",
    "severite": "orange"
}
```

**Gold — enrichie + analysée par l'agent :**
```
• Type : Flood crisis
• Sévérité : Orange (moins grave que 2021, attention immédiate requise)
• Actions recommandées : évacuation d'urgence, déploiement des équipes,
  restauration des services essentiels — basé sur crises passées similaires

Latence : ~1 450 ms | Tokens : 362 | Tracé dans Langfuse + MLflow
```

---

## 🤖 Agent LangGraph

| Node | Rôle | Technologie |
|---|---|---|
| **Retrieve** | Cherche les crises passées similaires | ChromaDB + `all-MiniLM-L6-v2` |
| **Analyse** | Génère un rapport contextualisé | Groq / Llama 3.3 70B |
| **Rapport** | Retourne 3 bullet points structurés | LangGraph State |

**Pourquoi RAG ?**
- Sans RAG → le LLM répond depuis son entraînement général → risque d'hallucination
- Avec RAG → l'agent consulte d'abord les crises passées → réponse ancrée, traçable, vérifiable

---

## 📊 Data Pipeline — Medallion + Data Mesh

```
Bronze  →  données brutes ingérées (JSON, CSV) — stockage S3
Silver  →  données nettoyées (Pandas + Pydantic validation)
Gold    →  données enrichies + embeddings — prêtes pour le RAG
```

Chaque source (news, GIS, alertes) = domaine **Data Mesh** indépendant avec son propre pipeline.

---

## 🛠️ Stack technique

### LLM & Agents
- `langchain` + `langchain-groq` — orchestration LLM
- `langgraph` — orchestration multi-agents avec état
- `langfuse` — observabilité et tracing (latence, tokens, prompts)
- LLM : **Groq / Llama 3.3 70B**

### RAG & Embeddings
- `chromadb` — vector store local
- `sentence-transformers` (`all-MiniLM-L6-v2`) — embeddings
- Chunking stratégique + scoring de similarité

### Data Pipeline
- `pandas` — nettoyage et transformation (Bronze → Silver)
- `pydantic` — validation stricte des types (PEP 484)
- `boto3` — interface S3 (Bronze/Silver/Gold)
- `python-dotenv` — gestion des secrets

### MLOps
- `mlflow` — tracking expériences (paramètres, métriques, runs)
- `docker` — conteneurisation
- **GitHub Actions** — CI/CD (black + pytest à chaque push)

### API & Sécurité
- `fastapi` + `uvicorn` — API REST asynchrone
- `APIKeyHeader` — authentification par clé (`X-API-Key`)
- Sans clé → **401 Unauthorized** | Avec clé → **200 OK**
- Documentation Swagger auto : `/docs`

### Cloud AWS
| Service | Usage |
|---|---|
| **ECR** | Registre Docker |
| **ECS Fargate** | Exécution sans serveur |
| **S3** | Stockage Bronze/Silver/Gold |
| **CloudWatch** | Logs et monitoring |

### Qualité du code Python
- **PEP 8** — formatage (`black`)
- **PEP 257** — docstrings (Args/Returns)
- **PEP 484** — type hints (`mypy`)
- **pytest** — 3 tests passent en 0.03s ✅

---

## 📁 Structure du projet

```
crisis-intel-agent/
│
├── agents/
│   ├── __init__.py
│   └── scout_agent.py        # Agent LangGraph (Retrieve → Analyse → Rapport)
├── api/
│   ├── __init__.py
│   └── main.py               # FastAPI — /health + /ingest sécurisé
├── data/
│   ├── bronze/
│   │   ├── alerts.csv
│   │   ├── alerts.json
│   │   └── storage.py        # Client S3 local
│   ├── silver/
│   │   ├── clean.py          # clean_text + clean_alert
│   │   ├── models.py         # Alert — modèle Pydantic
│   │   └── transform.py      # Pipeline Bronze → Silver
│   └── gold/
│       └── crisis_reports.py # Base de connaissances RAG
├── evaluation/
│   ├── __init__.py
│   └── vector_store.py       # ChromaDB — build + search
├── mlflow_tracking/
│   ├── __init__.py
│   └── tracker.py            # MLflow — log params + metrics
├── infrastructure/
│   ├── docker/
│   │   ├── Dockerfile
│   │   └── task-definition.json
│   └── k8s/
├── tests/
│   ├── __init__.py
│   └── test_clean.py         # 3 tests pytest
├── .github/
│   └── workflows/
│       └── ci.yml            # GitHub Actions CI
├── .gitignore
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🚀 Lancement rapide (local)

```bash
# Cloner
git clone https://github.com/FatimaChahal/crisis-intel-agent
cd crisis-intel-agent

# Environnement virtuel
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Variables d'environnement
cp .env.example .env
# Remplir : GROQ_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, API_KEY

# Lancer l'API
uvicorn api.main:app --reload
# http://localhost:8000/docs
```

## 🧪 Tests

```bash
python -m pytest tests/ -v
# 3 passed in 0.03s ✅
```

## 🤖 Lancer l'agent

```bash
python -m agents.scout_agent
```

---

## ☁️ Déploiement AWS

```bash
# Login ECR
aws ecr get-login-password --region eu-west-1 | \
  docker login --username AWS --password-stdin \
  637161850292.dkr.ecr.eu-west-1.amazonaws.com

# Build + Push
docker build -t crisis-intel-agent -f infrastructure/docker/Dockerfile .
docker tag crisis-intel-agent:latest \
  637161850292.dkr.ecr.eu-west-1.amazonaws.com/crisis-intel-agent:latest
docker push \
  637161850292.dkr.ecr.eu-west-1.amazonaws.com/crisis-intel-agent:latest
```

🌍 **API Live** : `http://34.248.159.126:8000/docs`

---

## 📈 CI/CD — GitHub Actions

À chaque `git push` sur `main` :
1. ✅ Installation des dépendances
2. ✅ Vérification du style (`black --check`)
3. ✅ Lancement des tests (`pytest`)

---

## 🔐 Sécurité

- Clé API obligatoire dans le header `X-API-Key`
- Secrets dans `.env` — jamais dans le code
- `.env` exclu de Git via `.gitignore`
- Security Group AWS — port 8000 uniquement

---

## 👩‍💻 Auteur

**Fatima Chahal** — AI Engineer | PhD in Distributed Systems (UTT)  
🔗 [GitHub](https://github.com/FatimaChahal) · [Google Scholar](https://scholar.google.com/citations?user=I106NZcAAAAJ&hl=fr)