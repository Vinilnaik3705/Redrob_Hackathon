import json
import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
from scorer import (
    check_honeypots, check_disqualifications, clean_text,
    score_career_retrieval, score_production_deployment,
    score_behavioral_availability, score_skill_match,
    score_rules_context, score_yoe, calculate_penalties,
    score_assessment, score_career_trajectory
)

candidates_path = "candidates.jsonl"
out_dir = "."

if not os.path.exists(candidates_path):
    candidates_path = r"c:\Users\VINIL NAIK\OneDrive\Desktop\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"

if not os.path.exists(candidates_path):
    print(f"Error: candidates.jsonl not found at {candidates_path}")
    sys.exit(1)

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loading Sentence Transformer model (BAAI/bge-small-en-v1.5) on {device}...")
model = SentenceTransformer('BAAI/bge-small-en-v1.5', device=device)

candidate_ids = []
texts = []
features = []
honeypots = []

print("Streaming candidates.jsonl and precomputing features...")
count = 0

with open(candidates_path, "r", encoding="utf-8") as f:
    for line_idx, line in enumerate(f):
        if not line.strip():
            continue
        try:
            c = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Warning: Skipped invalid JSON line {line_idx+1}: {e}")
            continue
        
        cid = c["candidate_id"]
        profile = c.get("profile", {})
        headline = profile.get("headline", "")
        summary = profile.get("summary", "")
        yoe = profile.get("years_of_experience", 0.0)
        
        career_texts = []
        for job in c.get("career_history", []):
            title = job.get("title", "")
            desc = job.get("description", "")
            company = job.get("company", "")
            career_texts.append(f"{title} at {company}: {desc}")
        career_history_str = " ".join(career_texts)
        
        skills_list = [s.get("name", "") for s in c.get("skills", [])]
        skills_str = " ".join(skills_list)
        
        full_text = clean_text(f"{headline} {summary} {career_history_str} {skills_str}")
        texts.append(full_text)
        
        is_bad = check_honeypots(c) or check_disqualifications(c)
        
        retrieval_score = score_career_retrieval(c)
        production_score = score_production_deployment(c)
        skill_score = score_skill_match(c)
        behavioral_score = score_behavioral_availability(c)
        rules_score = score_rules_context(c)
        yoe_score = score_yoe(yoe)
        penalties_total = calculate_penalties(c)
        assessment_score = score_assessment(c)
        trajectory_score = score_career_trajectory(c)
        
        candidate_ids.append(cid)
        features.append([
            retrieval_score, 
            production_score, 
            skill_score, 
            behavioral_score, 
            rules_score, 
            yoe_score, 
            penalties_total,
            assessment_score,
            trajectory_score
        ])
        honeypots.append(is_bad)
        
        count += 1
        if count % 10000 == 0:
            print(f"Processed {count} profiles...")

print(f"Total candidates processed: {count}")

print("Batch encoding texts with SentenceTransformer (batch_size=256)...")
embeddings = model.encode(texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True)

print("Building BM25 index...")
from rank_bm25 import BM25Okapi
import pickle
tokenized_corpus = [text.lower().split() for text in texts]
bm25 = BM25Okapi(tokenized_corpus)

print("Saving precomputed data to disk...")
np.save(os.path.join(out_dir, "embeddings.npy"), embeddings)
np.save(os.path.join(out_dir, "features.npy"), np.array(features, dtype=np.float32))
np.save(os.path.join(out_dir, "honeypots.npy"), np.array(honeypots, dtype=bool))

with open(os.path.join(out_dir, "ids.json"), "w", encoding="utf-8") as out_f:
    json.dump(candidate_ids, out_f)

with open(os.path.join(out_dir, "bm25_index.pkl"), "wb") as f:
    pickle.dump({"bm25": bm25, "tokenized_corpus": tokenized_corpus}, f)

print("Precomputation finished successfully.")
