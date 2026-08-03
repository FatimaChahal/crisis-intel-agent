# 🔥 Crisis Intel Agent — Wildfire Intelligence Platform

> **Plateforme agentique de veille et d'analyse des incendies de forêt en Europe**  
> Multi-agent · Agentic RAG · MLOps · Cloud AWS · API REST · CI/CD · Évaluation RAG

---

## 🎯 Problématique

Les opérateurs de gestion de crise (pompiers, préfectures, ONG, protection civile) manquent d'un outil pour **accéder instantanément aux leçons tirées des incendies passés similaires** lors d'une situation d'urgence.

**Sans système :**
```
Opérateur cherche manuellement dans des archives PDF, Excel, rapports...
→ Des heures perdues | Risque de rater des infos cruciales | Chaque minute compte
```

**Avec Crisis Intel Agent :**
```
Opérateur : "Y a-t-il un risque d'incendie aujourd'hui en Gironde ?"
→ Météo temps réel : 35°C, humidité 28%, risque 🟠 HIGH
→ Cas similaires : Gironde 2022, Bouches-du-Rhône 2017...
→ Actions recommandées basées sur données réelles Copernicus
→ Réponse en < 1 seconde, dans la langue de l'opérateur
```

---

## 📊 Données réelles — Copernicus EFFIS + MODIS

| Source | Description | Volume |
|---|---|---|
| **EFFIS Copernicus** | Totaux par pays/année 1980-2024 | 45 années × 31 pays |
| **MODIS Satellite** | Incendies individuels avec polygones GPS | 102 561 incendies |
| **Silver** | Données nettoyées + features calculées | 69 435 lignes |
| **Gold** | Enrichi + risk_score + summaries textuels | 18 607 incendies |

---

## 🤖 Architecture des 4 agents LangGraph

```
Question utilisateur
      │
      ▼
CLASSIFIER ──▶ RETRIEVER ──▶ ANALYST ──▶ RESPONDER
Langue/type    ChromaDB      Stats         [1][2][3]
Guardrails     Validation    Patterns      Multilingue
      │
      ├── MCP Weather Tool (OpenMeteo — temps réel)
```

| Agent | Rôle |
|---|---|
| **CLASSIFIER** | Langue + type + guardrail + trigger météo |
| **RETRIEVER** | ChromaDB search + validation qualité |
| **ANALYST** | Stats, patterns, synthèse structurée |
| **RESPONDER** | Réponse finale + citations [1][2][3] + météo |

---

## 📐 Évaluation RAG — 45 questions professionnelles

| Métrique | Score | Seuil pro | Status |
|---|---|---|---|
| **Guardrail Precision** | 0.90 | ≥ 0.85 | ✅ |
| **Wildfire Recall** | 1.00 | ≥ 0.85 | ✅ |
| **Off-topic Block Rate** | 0.80 | ≥ 0.80 | ✅ |
| **Avg Faithfulness** | 0.96 | ≥ 0.85 | ✅ |
| **Avg Relevancy** | 0.95 | ≥ 0.85 | ✅ |
| **Avg Latency** | 792ms | ≤ 2000ms | ✅ |

---

## 🛠️ Stack technique

- **LLM** : Groq / Llama 3.3 70B
- **Agents** : LangChain + LangGraph
- **RAG** : ChromaDB + all-MiniLM-L6-v2
- **MCP** : OpenMeteo API (météo temps réel)
- **MLOps** : MLflow + Langfuse + Docker
- **Interface** : Gradio (chatbox multilingue)
- **API** : FastAPI + API Key security
- **Cloud** : AWS ECR + ECS Fargate
- **CI/CD** : GitHub Actions (black + pytest)
- **Data** : Pandas + GeoPandas + Pydantic

---

## 🚀 Lancement rapide

```bash
git clone https://github.com/FatimaChahal/crisis-intel-agent
cd crisis-intel-agent
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Remplir GROQ_API_KEY, LANGFUSE_*, API_KEY

# Pipeline Medallion
python3 data/bronze/ingest.py
python3 data/silver/transform_fires.py
python3 data/gold/enrich_fires.py
python3 -m evaluation.vector_store

# Chatbox
python3 -m api.chatbox  # → http://localhost:7860

# Évaluation
python3 -m evaluation.full_eval
```

---

## 🌍 API Live

`http://34.248.159.126:8000/docs`

---

## 👩‍💻 Auteur

**Fatima Chahal** — AI Engineer | PhD in Distributed Systems (UTT)  
🔗 [GitHub](https://github.com/FatimaChahal) · [Google Scholar](https://scholar.google.com/citations?user=I106NZcAAAAJ&hl=fr)