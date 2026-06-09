import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scorer import score_candidate, is_consulting, JD_SKILLS

# Fixed Job Description Query
JD_QUERY = (
    "Senior AI Engineer. Production experience with embeddings-based retrieval systems "
    "(sentence-transformers, OpenAI embeddings, BGE, E5), vector databases or hybrid search "
    "infrastructure (Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS), Python, "
    "evaluation frameworks for ranking systems (NDCG, MRR, MAP, offline-to-online correlation, A/B testing). "
    "LLM fine-tuning (LoRA, QLoRA, PEFT), learning-to-rank. Startup product company experience. "
    "Located in Noida or Pune, India."
)

def generate_reasoning(candidate, score, rank):
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    
    yoe = profile.get("years_of_experience", 0.0)
    title = profile.get("current_title", "Engineer")
    
    # Check for product company roles
    product_companies = []
    for job in candidate.get("career_history", []):
        comp = job.get("company", "")
        ind = job.get("industry", "")
        if comp and not is_consulting(comp, ind):
            if comp not in product_companies:
                product_companies.append(comp)
                
    company_type = "product company" if product_companies else "services firm"
    company_detail = f"at {company_type}"
    if product_companies:
        company_detail += f" ({', '.join(product_companies[:2])})"
    
    # Find matching skills
    skills_list = [s.get("name", "") for s in candidate.get("skills", [])]
    matched = [s for s in skills_list if s.lower() in JD_SKILLS]
    if matched:
        skills_str = f"expertise in {', '.join(matched[:3])}"
    else:
        skills_str = f"skills in {', '.join(skills_list[:3])}"
        
    notice = signals.get("notice_period_days", 180)
    last_active = signals.get("last_active_date", "")
    active_days_ago = None
    if last_active:
        try:
            today = datetime(2026, 6, 7)
            active_date = datetime.strptime(last_active, "%Y-%m-%d")
            active_days_ago = (today - active_date).days
        except:
            pass
            
    if active_days_ago is not None:
        if active_days_ago <= 30:
            availability = "recently active on platform"
        elif active_days_ago <= 90:
            availability = f"active {active_days_ago} days ago"
        else:
            availability = "inactive"
    else:
        availability = "active status unknown"
        
    # Diverse templates based on rank and profile data
    if rank <= 10:
        reasoning = (
            f"Top tier candidate with {yoe:.1f} years as {title} {company_detail}. "
            f"Strong matching {skills_str}. Profile is {availability} with a notice period of {notice} days. "
            f"Strong fit on retrieval and evaluation requirements."
        )
    elif rank <= 30:
        reasoning = (
            f"Highly relevant {yoe:.1f}-year {title} {company_detail}. "
            f"Possesses {skills_str}; {availability} with notice period of {notice} days."
        )
    else:
        reasoning = (
            f"Qualified {title} with {yoe:.1f} years experience {company_detail}. "
            f"Offers {skills_str}. {availability}, {notice}d notice."
        )
    return reasoning

def main():
    parser = argparse.ArgumentParser(description="Rank candidates for Senior AI Engineer.")
    parser.add_argument("--candidates", required=True, help="Path to candidates json/jsonl file.")
    parser.add_argument("--out", required=True, help="Path to write the submission CSV.")
    args = parser.parse_args()
    
    start_time = datetime.now()
    
    print(f"Loading candidates from {args.candidates}...")
    candidates = []
    if args.candidates.endswith(".json"):
        with open(args.candidates, "r", encoding="utf-8") as f:
            candidates = json.load(f)
    else:
        with open(args.candidates, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    candidates.append(json.loads(line))
                    
    print(f"Loaded {len(candidates)} candidates.")
    
    # Paths for precomputed files (check script dir, then fallback to current dir)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    embeddings_path = os.path.join(base_dir, "embeddings.npy")
    if not os.path.exists(embeddings_path):
        embeddings_path = "embeddings.npy"
        
    features_path = os.path.join(base_dir, "features.npy")
    if not os.path.exists(features_path):
        features_path = "features.npy"
        
    honeypots_path = os.path.join(base_dir, "honeypots.npy")
    if not os.path.exists(honeypots_path):
        honeypots_path = "honeypots.npy"
        
    ids_path = os.path.join(base_dir, "ids.json")
    if not os.path.exists(ids_path):
        ids_path = "ids.json"
    
    use_precomputed = False
    
    if os.path.exists(embeddings_path) and os.path.exists(features_path) and os.path.exists(honeypots_path) and os.path.exists(ids_path):
        print("Precomputed files found. Checking if they match input candidates...")
        with open(ids_path, "r", encoding="utf-8") as f:
            precomputed_ids = json.load(f)
            
        if len(candidates) == len(precomputed_ids):
            # Check ID order match
            id_match = True
            for i in range(min(100, len(candidates))):
                if candidates[i]["candidate_id"] != precomputed_ids[i]:
                    id_match = False
                    break
            if id_match:
                use_precomputed = True
                print("Using precomputed features and embeddings for fast ranking.")
                
    if use_precomputed:
        pre_embeddings = np.load(embeddings_path)
        pre_features = np.load(features_path)
        pre_honeypots = np.load(honeypots_path)
        
        # Load Sentence Transformer to embed the JD
        print("Embedding job description...")
        model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        jd_vector = model.encode([JD_QUERY])[0]
        
        print("Computing cosine similarity...")
        semantic_scores = cosine_similarity([jd_vector], pre_embeddings)[0]
        
        print("Computing final scores...")
        scores = []
        for i in range(len(candidates)):
            if pre_honeypots[i]:
                scores.append(0.0)
            else:
                scores.append(score_candidate(candidates[i], semantic_scores[i]))
    else:
        print("Computing features and embeddings dynamically (fallback)...")
        # Load Sentence Transformer model
        model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        
        # Build texts for candidates
        texts = []
        for c in candidates:
            profile = c.get("profile", {})
            headline = profile.get("headline", "")
            summary = profile.get("summary", "")
            career_texts = []
            for job in c.get("career_history", []):
                career_texts.append(f"{job.get('title', '')} at {job.get('company', '')}: {job.get('description', '')}")
            skills_str = " ".join([s.get("name", "") for s in c.get("skills", [])])
            full_text = f"{headline} {summary} {' '.join(career_texts)} {skills_str}"
            texts.append(full_text)
            
        print("Embedding candidates...")
        cand_embeddings = model.encode(texts, batch_size=256, convert_to_numpy=True)
        
        print("Embedding job description...")
        jd_vector = model.encode([JD_QUERY])[0]
        semantic_scores = cosine_similarity([jd_vector], cand_embeddings)[0]
        
        print("Scoring candidates...")
        scores = []
        for i, c in enumerate(candidates):
            scores.append(score_candidate(c, semantic_scores[i]))
            
    # Sort and rank
    print("Ranking candidates...")
    ranked_list = []
    for i, c in enumerate(candidates):
        ranked_list.append({
            "candidate_id": c["candidate_id"],
            "score": scores[i],
            "candidate": c
        })
        
    # Tie-breaking: score descending (rounded to 4 decimals), then candidate_id ascending
    ranked_list.sort(key=lambda x: (-round(x["score"], 4), x["candidate_id"]))
    
    # Select top 100
    top_100 = ranked_list[:100]
    
    # Generate CSV content
    rows = []
    for rank_idx, item in enumerate(top_100):
        rank = rank_idx + 1
        cid = item["candidate_id"]
        score = item["score"]
        reason = generate_reasoning(item["candidate"], score, rank)
        rows.append({
            "candidate_id": cid,
            "rank": rank,
            "score": round(score, 4),
            "reasoning": reason
        })
        
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False, columns=["candidate_id", "rank", "score", "reasoning"])
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"Ranking completed in {duration:.2f} seconds. Output saved to {args.out}")

if __name__ == "__main__":
    main()
