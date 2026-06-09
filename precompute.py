import json
import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
from scorer import (
    calculate_rules_score, calculate_skill_match_score, calculate_behavioral_score, 
    is_honeypot, clean_text, check_career_for_production, check_career_for_retrieval, 
    is_title_chaser, is_research_only, is_langchain_only, is_closed_source_only
)

candidates_path = "candidates.jsonl"
out_dir = "."

if not os.path.exists(candidates_path):
    # Fallback to local Windows path
    candidates_path = r"c:\Users\VINIL NAIK\OneDrive\Desktop\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"
    out_dir = r"c:\Users\VINIL NAIK\OneDrive\Desktop\[PUB] India_runs_data_and_ai_challenge\hackathon"

if not os.path.exists(candidates_path):
    print(f"Error: candidates.jsonl not found at {candidates_path}")
    sys.exit(1)

os.makedirs(out_dir, exist_ok=True)

print("Loading Sentence Transformer model (BAAI/bge-small-en-v1.5)...")
model = SentenceTransformer('BAAI/bge-small-en-v1.5')

candidate_ids = []
texts = []
features = []
honeypots = []

print("Streaming candidates.jsonl and extracting features...")
count = 0

with open(candidates_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        c = json.loads(line)
        
        cid = c["candidate_id"]
        profile = c.get("profile", {})
        headline = profile.get("headline", "")
        summary = profile.get("summary", "")
        
        career_texts = []
        for job in c.get("career_history", []):
            title = job.get("title", "")
            desc = job.get("description", "")
            company = job.get("company", "")
            career_texts.append(f"{title} at {company}: {desc}")
        career_history_str = " ".join(career_texts)
        
        skills_list = [s.get("name", "") for s in c.get("skills", [])]
        skills_str = " ".join(skills_list)
        
        # Build text for semantic embedding
        full_text = clean_text(f"{headline} {summary} {career_history_str} {skills_str}")
        texts.append(full_text)
        
        # Precompute rule, skill, and behavioral scores + honeypot flag
        rules_score = calculate_rules_score(c)
        
        # Skills match corroboration
        skill_score = calculate_skill_match_score(skills_list)
        career_has_retrieval = check_career_for_retrieval(c.get("career_history", []))
        if not career_has_retrieval:
            skill_score *= 0.3
            
        behavioral_score = calculate_behavioral_score(c)
        career_evidence = check_career_for_production(c.get("career_history", []))
        
        # Penalties calculation
        penalties = 0.0
        if is_title_chaser(c.get("career_history", [])):
            penalties += 0.15
        if is_research_only(c.get("career_history", [])):
            penalties += 0.15
        if is_langchain_only(c):
            penalties += 0.15
        if is_closed_source_only(c):
            penalties += 0.10
            
        honeypot_flag = is_honeypot(c)
        
        candidate_ids.append(cid)
        features.append([rules_score, skill_score, behavioral_score, career_evidence, penalties])
        honeypots.append(honeypot_flag)
        
        count += 1
        if count % 10000 == 0:
            print(f"Processed {count} profiles...")

print(f"Total candidates processed: {count}")

print("Batch encoding texts with SentenceTransformer (batch_size=256)...")
embeddings = model.encode(texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True)

# Save precomputed arrays
print("Saving precomputed data to disk...")
np.save(os.path.join(out_dir, "embeddings.npy"), embeddings)
np.save(os.path.join(out_dir, "features.npy"), np.array(features, dtype=np.float32))
np.save(os.path.join(out_dir, "honeypots.npy"), np.array(honeypots, dtype=bool))

with open(os.path.join(out_dir, "ids.json"), "w", encoding="utf-8") as out_f:
    json.dump(candidate_ids, out_f)

print("Precomputation finished successfully.")
