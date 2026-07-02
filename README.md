# 🚨 Crisis Intel Agent

> **Plateforme agentique de veille et d'analyse de crises géospatiales**  
> Multi-agent · Agentic RAG · MLOps · Cloud AWS · API REST

---

## 🎯 Objectif

Crisis Intel Agent est une plateforme IA bout-en-bout qui automatise la **veille, l'analyse et le résumé de situations de crise** (inondations, risques environnementaux, événements géospatiaux) à partir de sources de données hétérogènes (news, données GIS, capteurs).

Le système est construit autour d'une architecture **multi-agents orchestrée avec LangGraph**, d'un pipeline **Agentic RAG évalué**, et d'un déploiement **MLOps-grade sur AWS**.

---

## 🏗️ Architecture

```
Sources (News / GIS / Capteurs)
        │
        ▼
┌──────────────────────────────────────────┐
│           Data Pipeline (Medallion)       │
│  Bronze → Silver → Gold                  │
│  (raw)    (clean)   (enriched)           │
└──────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│         Agent Orchestrator (LangGraph)   │
│                                          │
│   ┌──────────┐   ┌──────────────────┐   │
│   │  Scout   │──▶│    Analyst       │   │
│   │  Agent   │   │    Agent         │   │
│   └──────────┘   └──────────────────┘   │
│         │               │               │
│         ▼               ▼               │
│   ┌──────────┐   ┌──────────────────┐   │
│   │   RAG    │   │    Critic        │   │
│   │  Agent   │   │    Agent         │   │
│   └──────────┘   └──────────────────┘   │
└──────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│         FastAPI REST API                 │
│  /analyze  /evaluate  /health            │
└──────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│     Observabilité & MLOps                │
│  Langfuse · MLflow · Prometheus          │
└──────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│     Cloud AWS (Docker + Kubernetes)      │
│  ECS Fargate / EKS · ECR · S3           │
└──────────────────────────────────────────┘
```

---

## 🤖 Agents

| Agent | Rôle | Technologie |
|---|---|---|
| **Scout Agent** | Collecte et récupère les documents pertinents (news, GIS, rapports) | LangChain + web search |
| **RAG Agent** | Interroge la base vectorielle, génère une réponse contextualisée | LangGraph + ChromaDB |
| **Analyst Agent** | Analyse et structure l'information en rapport de crise | LangGraph + LLM |
| **Critic Agent** | Vérifie la fiabilité et la cohérence du rapport généré | LangGraph + scoring |

Communication inter-agents : pattern **A2A (Agent-to-Agent)** via protocole **MCP (Model Context Protocol)**.

---

## 📊 Data Pipeline — Architecture Medallion

```
Bronze  →  données brutes ingérées (JSON, GeoJSON, CSV) — stockage S3
Silver  →  données nettoyées, normalisées, déduplicées
Gold    →  données enrichies, prêtes pour le RAG et l'analyse
```

Chaque domaine de données (news, GIS, capteurs) est traité comme un **Data Mesh** indépendant avec son propre pipeline et ses propres contrats de données.

---

## 📐 Evaluation RAG

Métriques implémentées avec **RAGAS** :

| Métrique | Description |
|---|---|
| Faithfulness | La réponse est-elle fidèle aux sources récupérées ? |
| Answer Relevancy | La réponse répond-elle à la question posée ? |
| Context Precision | Les documents récupérés sont-ils pertinents ? |
| Context Recall | Tous les documents utiles ont-ils été récupérés ? |

Tracking des expériences RAG avec **MLflow** (paramètres, métriques, versions de modèles).

---

## 🛠️ Stack technique

### LLM & Agents
- `langchain`: orchestration et chaînes LLM
- `langgraph`: orchestration multi-agents avec état
- `langfuse`: observabilité et tracing des LLM
- LLMs : Gemini 2.5 Flash / Claude / Ollama (local)

### RAG & Embeddings
- `chromadb`: vector store
- `ragas`: évaluation RAG
- Chunking stratégique + scoring de similarité

### MLOps
- `mlflow`: tracking des expériences, modèles, métriques
- `prometheus`: monitoring applicatif
- `docker`: conteneurisation
- `kubernetes` (EKS): orchestration

### API
- `fastapi`: API REST asynchrone
- `pydantic`: validation des données (PEP 484 — type hints)
- Endpoints RESTful documentés (OpenAPI/Swagger)

### Cloud AWS
| Service AWS | Usage |
|---|---|
| **ECS Fargate** | Exécution des conteneurs sans serveur |
| **EKS** | Orchestration Kubernetes |
| **ECR** | Registry des images Docker |
| **S3** | Stockage des données (Bronze/Silver/Gold) |
| **CloudWatch** | Monitoring et logs |

### CI/CD
- **GitHub Actions** → build Docker → push ECR → deploy ECS/EKS

### Qualité du code Python
- PEP 8: style et formatage (`black`, `flake8`)
- PEP 257: docstrings (`pydocstyle`)
- PEP 484: type hints (`mypy`)

---

## 📁 Structure du projet

```
crisis-intel-agent/
│
├── agents/                  # Agents LangGraph (Scout, RAG, Analyst, Critic)
├── api/                     # FastAPI — endpoints REST
├── data/
│   ├── bronze/              # Données brutes ingérées (S3)
│   ├── silver/              # Données nettoyées
│   └── gold/                # Données enrichies pour RAG
├── evaluation/              # Pipeline d'évaluation RAG (RAGAS)
├── infrastructure/
│   ├── docker/              # Dockerfiles
│   └── k8s/                 # Manifests Kubernetes (EKS)
├── mlflow/                  # Tracking MLflow
├── notebooks/               # Exploration et prototypage
├── tests/                   # Tests unitaires et d'intégration
├── docs/                    # Documentation architecture
├── .env.example
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🚀 Lancement rapide

```bash
# Cloner le repo
git clone https://github.com/FatimaChahal/crisis-intel-agent
cd crisis-intel-agent

# Variables d'environnement
cp .env.example .env

# Lancer avec Docker Compose (local)
docker-compose up --build

# API disponible sur http://localhost:8000
# Docs Swagger : http://localhost:8000/docs
```

---

## ☁️ Déploiement AWS

```bash
# Build et push vers ECR
docker build -t crisis-intel-agent .
aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.eu-west-1.amazonaws.com
docker tag crisis-intel-agent:latest <account>.dkr.ecr.eu-west-1.amazonaws.com/crisis-intel-agent:latest
docker push <account>.dkr.ecr.eu-west-1.amazonaws.com/crisis-intel-agent:latest

# Déploiement ECS Fargate via GitHub Actions (CI/CD automatisé)
```

---

## 🔗 Lien avec mes travaux de recherche

Ce projet prolonge directement mon travail postdoctoral sur **AI4MultiGIS** (plateforme IA pour la gestion de crises géospatiales, UPPA/LIUPPA) en appliquant les mêmes problématiques (données hétérogènes, gestion de crise, souveraineté) à une architecture **agentic AI production-grade sur AWS**.

---

## 📄 Statut

🚧 **En cours de développement**

| Composant | Statut |
|---|---|
| Structure du projet | ✅ |
| Data pipeline Medallion | 🔄 En cours |
| Agents LangGraph | 🔄 En cours |
| FastAPI REST | 🔄 En cours |
| Evaluation RAG (RAGAS) | ⬜ À venir |
| MLflow tracking | ⬜ À venir |
| Docker + EKS | ⬜ À venir |
| Déploiement AWS | ⬜ À venir |
| CI/CD GitHub Actions | ⬜ À venir |

---

## 👩‍💻 Auteur

**Fatima Chahal** - AI Engineer | PhD in Distributed Systems  
🔗 [GitHub](https://github.com/FatimaChahal) · [Google Scholar](https://scholar.google.com/citations?user=I106NZcAAAAJ&hl=fr)
