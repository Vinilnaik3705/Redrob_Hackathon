---
title: Intelligent Candidate Ranker
emoji: 🎖️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.35.0
python_version: 3.11
app_file: app.py
pinned: false
---

# Team The Gladiators

**Intelligent Candidate Ranking System**  
*India Runs Hackathon 2026 · Redrob AI · Track 1: Data & AI Challenge*

---

## Overview

We implement an advanced **4-stage candidate ranking pipeline** designed to identify the top 100 Senior AI Engineers from a pool of 100K+ candidates. Our approach combines semantic embeddings, career trajectory evidence, rule-based location/company preferences, and behavioral availability signals, while actively defending against keyword-stuffer traps and fake profile honeypots.

---

## Architecture

```
                 candidates.jsonl (100K)
                         │
                         ▼
     ┌──────────────────────────────────────┐
     │      Stage 1: Hard Knockouts         │  Eliminates honeypots, unrelated roles,
     │                                      │  consulting-only, inactive profiles
     └───────────────────┬──────────────────┘
                         │ (Survivors)
                         ▼
     ┌──────────────────────────────────────┐
     │ Stage 2: FAISS Semantic First-Pass   │  Filters pool down to top 3000
     │                                      │  using BAAI/bge-small-en-v1.5 embeddings
     └───────────────────┬──────────────────┘
                         │
                         ▼
     ┌──────────────────────────────────────┐
     │ Stage 3: Multi-Signal Full Scoring   │  Evaluates 6 dimensions and applies
     │                                      │  penalties
     └───────────────────┬──────────────────┘
                         │
                         ▼
     ┌──────────────────────────────────────┐
     │ Stage 4: Tie-Break & Output          │  Generates customized, fact-based
     │                                      │  reasoning strings
     └───────────────────┬──────────────────┘
                         │
                         ▼
            team_the gladiators.csv (top-100)
```

---

## Scoring Signals

### 6 Scoring Dimensions

| Dimension | Weight | What it captures |
|---|---|---|
| **Semantic Similarity** | 30% | Cosine similarity against Job Description using `BAAI/bge-small-en-v1.5` |
| **Career Retrieval Evidence** | 25% | Search/retrieval systems keywords frequency in career history |
| **Production Deployment Evidence** | 20% | Production, deployment, scaling, and runtime infrastructure keyword frequency |
| **Behavioral Availability** | 15% | Recency on platform, notice period, response rate, and interview completion rate |
| **Skill Match + Corroboration** | 7% | Required skill coverage verified against career history text & endorsements |
| **Rules / Context Fit** | 3% | Prefers product company history, preferred target cities, relocation interest, and GitHub |

### Penalty Multipliers & Hard Caps

| Condition | Penalty / Action |
|---|---|
| **Zero Retrieval Career Evidence** | Hard cap candidate final score to max of `0.25` |
| **Title Chaser** | Deduct `0.12` from final score (frequent short tenures) |
| **LangChain Only** | Deduct `0.10` from final score (lacks fundamental ML/NLP/IR skills) |
| **Research Only** | Deduct `0.15` from final score (no production or deployment signals) |
| **Junior Research Engineer** | Deduct `0.25` from final score (title is research and YOE < 5) |
| **Notice Period 31-60 Days** | Deduct `0.04` from final score |
| **Notice Period 61-90 Days** | Deduct `0.08` from final score |
| **Notice Period > 90 Days** | Deduct `0.12` from final score |
| **Pure Consulting Career** | Deduct `0.20` from final score (100% of career history at consulting firms) |
| **Closed-Source Only** | Deduct `0.08` from final score (YOE $\ge$ 5 and no GitHub) |
| **CV/Speech/Robotics Primary** | Deduct `0.10` from final score (majority career in CV/speech/robotics without NLP/IR) |

### Honeypot & Knockout Rules (Stage 1)

1. **Company Lifespan Violation** — Start date at company is earlier than the company founded year.
2. **Fake Profile Mismatch** — Junior title with senior YOE ($\ge$ 5 years).
3. **Fictional Company Trap** — Any career company matches fictional names (Hooli, Pied Piper, Globex, Initech, Dunder Mifflin, etc.).
4. **Expert Skills with Zero Endorsements / Duration** — $\ge$ 8 Expert-level skills with exactly 0 endorsements or 0 duration.
5. **HR/Marketing Title with ML Skills** — marketing/hr manager title containing ML skills.
6. **Too-Perfect Profile** — $\ge$ 3 simultaneous top-percentile flags (perfect response rate, perfect interview rate, round YOE + high assessments, etc.).
7. **All Skill Assessments Perfect** — $\ge$ 3 assessments, all scores $\ge$ 98.
8. **Unrelated Titles** — Disqualifies non-technical roles (HR, Project/Marketing Managers, sales, mechanical, etc.).
9. **Outside India** — Candidates residing outside of India.
10. **Inactive Profiles** — Candidates inactive on the platform for over 1 year.
11. **YOE Hard Minimum** — Experience below 3.0 years.

---

## Compute Constraints Compliance

| Constraint | Limit | The Gladiators Pipeline |
|---|---|---|
| **Runtime (cached)** | < 60 sec | **~50 seconds** on CPU (using precomputed embeddings) |
| **Runtime (cold)** | < 5 min | **~3-4 minutes** on CPU |
| **Peak Memory** | < 4 GB | ~2.5 GB peak memory |
| **Network during ranking** | Zero | Fully offline (no external APIs or network calls during evaluation) |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Precompute embeddings (One-time, ~20 min on CPU / ~5 min on GPU)

Generates local cached vector files (`embeddings.npy`, `features.npy`, `honeypots.npy`, and `ids.json`) to allow sub-minute execution on CPU:
```bash
python precompute.py
```

---

## Run

### 1. Generate ranked output

```bash
python rank.py --candidates ./candidates.jsonl --out ./team_the_gladiators.csv
```

### 2. Validate your submission

```bash
python validate_submission.py "team_the_gladiators.csv"
```

---

## Project Structure

```
redrob_hackathon/
├── jd_config.py                # JD constants: required skills, keywords, consulting companies, fictional companies
├── scorer.py                   # Core scoring logic, penalty functions, and honeypot detection
├── precompute.py               # Generates embeddings and extracts features (run once)
├── rank.py                     # Main entrypoint: load precomputed → FAISS pass → score → output CSV
├── validate_submission.py      # Format validator (included for convenience)
├── requirements.txt            # Project dependencies with exact version numbers
├── submission_metadata.yaml    # Portal metadata matching details
├── team_the gladiators.csv     # Final ranked top-100 submission CSV
└── .gitignore                  # Correctly configured git ignore patterns
```

---

## Tech Stack

| Component | Technology |
|---|---|
| **Language** | Python 3.9+ |
| **Embeddings** | `sentence-transformers` (`BAAI/bge-small-en-v1.5`) |
| **Vector Space & Indexing** | `faiss-cpu` (IndexFlatIP) |
| **Data Processing** | `pandas`, `numpy`, `scikit-learn` |
| **Format** | `pyyaml` (for metadata validation) |

---

## Author

**Vinil Naik**  
India Runs Hackathon 2026 · Team: **The Gladiators**  
GitHub: [Vinilnaik3705](https://github.com/Vinilnaik3705)
