# Team Antigravity — India Runs Data & AI Challenge

## Approach
Hybrid candidate ranking pipeline combining semantic embeddings 
(BAAI/bge-small-en-v1.5) with career evidence scoring, rule-based 
signals, and behavioral availability signals.

**Pipeline:**
1. Hard filter — eliminates honeypots, unrelated roles, consulting-only careers
2. Semantic similarity — BGE embeddings vs JD embedding
3. Career evidence — production + retrieval signals in job descriptions
4. Rule scoring — YOE range, product company, India location
5. Behavioral signals — notice period, platform activity, response rate

## How to Run

### Step 1 — Install dependencies
pip install -r requirements.txt

### Step 2 — Precompute embeddings (run once, ~20 min on GPU)
python precompute.py

### Step 3 — Generate ranked output
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

## Requirements
- Python 3.9+
- 16GB RAM
- GPU recommended for precompute step

## Output
CSV with columns: candidate_id, rank, score, reasoning (top 100 candidates)
