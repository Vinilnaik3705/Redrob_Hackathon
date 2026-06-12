import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import faiss
import torch
from datetime import datetime
from sentence_transformers import SentenceTransformer
from scorer import (
    check_honeypots, check_disqualifications, clean_text,
    score_career_retrieval, score_production_deployment,
    score_behavioral_availability, score_skill_match,
    score_rules_context, score_yoe, calculate_penalties,
    is_consulting, JD_REQUIRED_SKILLS
)

# Fixed Job Description Query
JD_QUERY = (
    "Senior AI Engineer with production experience in embeddings-based retrieval systems "
    "(sentence-transformers, BGE, E5, OpenAI embeddings), vector databases and hybrid search "
    "(Pinecone, Weaviate, Qdrant, Milvus, FAISS, Elasticsearch, OpenSearch), Python, "
    "ranking evaluation frameworks (NDCG, MRR, MAP, A/B testing, offline-to-online correlation). "
    "LLM fine-tuning (LoRA, QLoRA, PEFT), learning-to-rank. "
    "Startup product company experience. Located in India (Noida, Pune, Hyderabad, Bangalore, Mumbai, Delhi NCR)."
)

def generate_reasoning(candidate, score, rank):
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])
    
    yoe = profile.get("years_of_experience", 0.0)
    title = profile.get("current_title", "Engineer")
    notice = signals.get("notice_period_days", 180)
    
    # ALWAYS use ACTUAL companies from career_history — never filter
    # Show top 2 actual companies regardless of type
    all_companies = []
    seen = set()
    for job in career:
        comp = job.get("company", "").strip()
        if comp and comp not in seen:
            all_companies.append(comp)
            seen.add(comp)
            
    # Classify background honestly
    consulting_names = {
        "tcs", "tata consultancy", "infosys", "wipro", "accenture",
        "cognizant", "capgemini", "tech mahindra", "hcl", "hcltech",
        "deloitte", "kpmg", "ey", "ernst", "pwc", "mindtree",
        "mu sigma", "fractal analytics", "mphasis", "hexaware", "ltimindtree"
    }
    
    def is_consulting_company(name):
        return any(c in name.lower() for c in consulting_names)
        
    product_cos = [c for c in all_companies if not is_consulting_company(c)]
    consulting_cos = [c for c in all_companies if is_consulting_company(c)]
    
    # Honest background label
    if not consulting_cos:
        background = "product company"
    elif not product_cos:
        background = "services firm"
    else:
        background = "mixed background"  # Has both — be honest
        
    # Show actual companies (top 2) — NEVER substitute or filter
    display_companies = all_companies[:2]
    company_detail = f"at {background} ({', '.join(display_companies)})" if display_companies else f"at {background}"
    
    # Skills — only from actual skills[] array
    skills_list = [s.get("name", "") for s in candidate.get("skills", [])]
    matched = [s for s in skills_list if s.lower() in JD_REQUIRED_SKILLS]
    if matched:
        skills_str = f"expertise in {', '.join(matched[:3])}"
    else:
        # Fall back to top actual skills — never invent
        skills_str = f"skills including {', '.join(skills_list[:3])}" if skills_list else "general ML skills"
        
    # Availability — from actual signals
    last_active = signals.get("last_active_date", "")
    active_days_ago = None
    if last_active:
        try:
            today = datetime(2026, 6, 7)
            active_days_ago = (today - datetime.strptime(last_active, "%Y-%m-%d")).days
        except:
            pass
            
    if active_days_ago is not None:
        if active_days_ago <= 30:
            availability = "recently active"
        elif active_days_ago <= 90:
            availability = f"active {active_days_ago} days ago"
        elif active_days_ago <= 180:
            availability = f"last active {active_days_ago} days ago"
        else:
            availability = "inactive for 6+ months"
    else:
        last_days = signals.get("last_active_days_ago")
        if last_days is not None:
            if last_days <= 30:
                availability = "recently active"
            elif last_days <= 90:
                availability = f"active {last_days} days ago"
            else:
                availability = f"inactive {last_days} days"
        else:
            availability = "activity unknown"
            
    # Honest concerns — pull from actual data
    concerns = []
    if notice > 90:
        concerns.append(f"long notice period ({notice}d)")
    if "research" in title.lower() and not any(
        kw in " ".join(j.get("description","") for j in career).lower()
        for kw in ["production","deployed","shipped","scale"]
    ):
        concerns.append("research-focused background — production depth unclear")
    if consulting_cos and not product_cos:
        concerns.append("consulting-only background")
    if yoe > 12:
        concerns.append("senior tenure — verify active coding role")
    if yoe < 4:
        concerns.append("below ideal experience range")
        
    concern_str = f" Concern: {'; '.join(concerns[:2])}." if concerns else ""
    
    # Build reasoning with UNIQUE elements per candidate
    if rank <= 10:
        reasoning = (
            f"{yoe:.1f}-year {title} {company_detail}. "
            f"Strong {skills_str}; {availability}, notice {notice}d.{concern_str}"
        )
    elif rank <= 30:
        reasoning = (
            f"{yoe:.1f}-year {title} {company_detail}. "
            f"Relevant {skills_str}; {availability}, {notice}d notice.{concern_str}"
        )
    elif rank <= 60:
        reasoning = (
            f"{title} ({yoe:.1f}yr) {company_detail}. "
            f"{skills_str.capitalize()}; {availability}.{concern_str}"
        )
    else:
        reasoning = (
            f"{yoe:.1f}-year {title} {company_detail}. "
            f"{skills_str.capitalize()}; {notice}d notice, {availability}.{concern_str}"
        )
        
    return reasoning

def make_unique_reasonings(rows):
    seen = {}
    for row in rows:
        base = row['reasoning']
        if base in seen:
            sig = row.get('_signals', {})
            response = sig.get('recruiter_response_rate', 0)
            github = sig.get('github_activity_score', 0)
            open_flag = sig.get('open_to_work_flag', False)
            
            if open_flag:
                extra = " Actively seeking new opportunities."
            elif github and github > 0:
                extra = f" GitHub activity score: {github:.0f}."
            elif response:
                extra = f" Platform response rate: {response:.0%}."
            else:
                extra = f" Ranked #{row['rank']} by composite score."
                
            row['reasoning'] = base + extra
        seen[base] = True
    return rows

def main():
    parser = argparse.ArgumentParser(description="Rank candidates for Senior AI Engineer.")
    parser.add_argument("--candidates", required=True, help="Path to candidates json/jsonl file.")
    parser.add_argument("--out", required=True, help="Path to write the submission CSV.")
    parser.add_argument("--jd", required=False, help="Custom Job Description query text.")
    parser.add_argument("--limit", type=int, default=100, help="Number of top candidates to rank.")
    args = parser.parse_args()
    
    jd_query = args.jd if args.jd else JD_QUERY
    
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
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    embeddings_path = os.path.join(base_dir, "embeddings.npy")
    features_path = os.path.join(base_dir, "features.npy")
    honeypots_path = os.path.join(base_dir, "honeypots.npy")
    ids_path = os.path.join(base_dir, "ids.json")
    
    use_precomputed = False
    
    if os.path.exists(embeddings_path) and os.path.exists(features_path) and os.path.exists(honeypots_path) and os.path.exists(ids_path):
        print("Precomputed files found. Checking compatibility...")
        with open(ids_path, "r", encoding="utf-8") as f:
            precomputed_ids = json.load(f)
            
        if len(candidates) == len(precomputed_ids):
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
        
        # Build FAISS index of survivors (non-honeypot, non-disqualified)
        survivor_indices = np.where(~pre_honeypots)[0]
        
        scored_candidates = []
        if len(survivor_indices) == 0:
            print("Warning: All candidates were disqualified in Stage 1!")
            for idx in range(len(candidates)):
                scored_candidates.append({
                    "candidate_id": candidates[idx]["candidate_id"],
                    "score": 0.0,
                    "candidate": candidates[idx]
                })
        else:
            survivor_embeddings = pre_embeddings[survivor_indices]
            
            print("Embedding job description...")
            model = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cuda' if torch.cuda.is_available() else 'cpu')
            jd_vector = model.encode([jd_query])[0].astype(np.float32)
            jd_vector = jd_vector.reshape(1, -1)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(survivor_embeddings)
            faiss.normalize_L2(jd_vector)
            
            # FAISS Index
            dimension = 384
            index = faiss.IndexFlatIP(dimension)
            index.add(survivor_embeddings)
            
            # Search top 3000 semantic matches
            k = min(3000, len(survivor_indices))
            D, I = index.search(jd_vector, k)
            
            # I[0] contains indices in survivor_embeddings, D[0] contains similarity scores
            for j in range(k):
                surv_idx = I[0][j]
                orig_idx = survivor_indices[surv_idx]
                semantic_score = float(D[0][j])
                
                # Load precomputed features
                feat = pre_features[orig_idx]
                retrieval_score = float(feat[0])
                production_score = float(feat[1])
                skill_score = float(feat[2])
                behavioral_score = float(feat[3])
                rules_score = float(feat[4])
                yoe_score = float(feat[5])
                penalties_total = float(feat[6])
                
                # Combine score
                final_score = (
                    0.30 * semantic_score +
                    0.25 * retrieval_score +
                    0.20 * production_score +
                    0.15 * behavioral_score +
                    0.07 * skill_score +
                    0.03 * rules_score -
                    penalties_total
                )
                final_score = max(0.0, final_score)
                
                # Override Hard Cap if zero retrieval evidence with graduated penalty
                if retrieval_score == 0.0:
                    # Semantic score still counts but heavily discounted
                    base = semantic_score * 0.25  # max 0.25 for zero retrieval
                    final_score = max(0.0, base)
                elif retrieval_score < 0.3:
                    # Weak retrieval — partial discount
                    multiplier = 0.5 + (retrieval_score / 0.3) * 0.3  # 0.5 to 0.8
                    final_score = final_score * multiplier
                    
                scored_candidates.append({
                    "candidate_id": candidates[orig_idx]["candidate_id"],
                    "score": final_score,
                    "candidate": candidates[orig_idx]
                })
                
            # Add all other candidates (disqualified + non-top 3000 survivors) with 0.0 score
            scored_ids = {x["candidate_id"] for x in scored_candidates}
            for idx in range(len(candidates)):
                cid = candidates[idx]["candidate_id"]
                if cid not in scored_ids:
                    scored_candidates.append({
                        "candidate_id": cid,
                        "score": 0.0,
                        "candidate": candidates[idx]
                    })
                    
        # Filter out zero-score candidates first
        scored_candidates = [x for x in scored_candidates if x["score"] > 0.0]
        # Sort and rank (tie-break: score desc, then candidate_id asc)
        scored_candidates.sort(key=lambda x: (-round(x["score"], 4), x["candidate_id"]))
        top_items = scored_candidates[:min(args.limit, len(scored_candidates))]
    else:
        print("Computing features and embeddings dynamically (fallback)...")
        model = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Run hard knockouts
        survivors = []
        survivor_orig_indices = []
        for idx, c in enumerate(candidates):
            if not check_honeypots(c) and not check_disqualifications(c):
                survivors.append(c)
                survivor_orig_indices.append(idx)
                
        print(f"Survivors after Stage 1 hard filters: {len(survivors)}")
        
        scored_candidates = []
        if len(survivors) == 0:
            for c in candidates:
                scored_candidates.append({
                    "candidate_id": c["candidate_id"],
                    "score": 0.0,
                    "candidate": c
                })
        else:
            # Build texts for survivors
            texts = []
            for c in survivors:
                profile = c.get("profile", {})
                headline = profile.get("headline", "")
                summary = profile.get("summary", "")
                career_texts = []
                for job in c.get("career_history", []):
                    career_texts.append(f"{job.get('title', '')} at {job.get('company', '')}: {job.get('description', '')}")
                skills_str = " ".join([s.get("name", "") for s in c.get("skills", [])])
                full_text = clean_text(f"{headline} {summary} {' '.join(career_texts)} {skills_str}")
                texts.append(full_text)
                
            print("Embedding survivors...")
            surv_embeddings = model.encode(texts, batch_size=256, convert_to_numpy=True).astype(np.float32)
            
            print("Embedding job description...")
            jd_vector = model.encode([jd_query])[0].astype(np.float32)
            jd_vector = jd_vector.reshape(1, -1)
            
            faiss.normalize_L2(surv_embeddings)
            faiss.normalize_L2(jd_vector)
            
            dimension = 384
            index = faiss.IndexFlatIP(dimension)
            index.add(surv_embeddings)
            
            k = min(3000, len(survivors))
            D, I = index.search(jd_vector, k)
            
            for j in range(k):
                surv_idx = I[0][j]
                c = survivors[surv_idx]
                semantic_score = float(D[0][j])
                
                # Dynamic feature extraction
                retrieval_score = score_career_retrieval(c)
                production_score = score_production_deployment(c)
                skill_score = score_skill_match(c)
                behavioral_score = score_behavioral_availability(c)
                rules_score = score_rules_context(c)
                penalties_total = calculate_penalties(c)
                
                final_score = (
                    0.30 * semantic_score +
                    0.25 * retrieval_score +
                    0.20 * production_score +
                    0.15 * behavioral_score +
                    0.07 * skill_score +
                    0.03 * rules_score -
                    penalties_total
                )
                final_score = max(0.0, final_score)
                
                if retrieval_score == 0.0:
                    # Semantic score still counts but heavily discounted
                    base = semantic_score * 0.25  # max 0.25 for zero retrieval
                    final_score = max(0.0, base)
                elif retrieval_score < 0.3:
                    # Weak retrieval — partial discount
                    multiplier = 0.5 + (retrieval_score / 0.3) * 0.3  # 0.5 to 0.8
                    final_score = final_score * multiplier
                    
                scored_candidates.append({
                    "candidate_id": c["candidate_id"],
                    "score": final_score,
                    "candidate": c
                })
                
            # Add all other candidates (disqualified + non-top 3000 survivors) with 0.0 score
            scored_ids = {x["candidate_id"] for x in scored_candidates}
            for c in candidates:
                if c["candidate_id"] not in scored_ids:
                    scored_candidates.append({
                        "candidate_id": c["candidate_id"],
                        "score": 0.0,
                        "candidate": c
                    })
                    
        scored_candidates = [x for x in scored_candidates if x["score"] > 0.0]
        scored_candidates.sort(key=lambda x: (-round(x["score"], 4), x["candidate_id"]))
        top_items = scored_candidates[:min(args.limit, len(scored_candidates))]
            
    # Output top items to CSV
    rows = []
    for rank_idx, item in enumerate(top_items):
        rank = rank_idx + 1
        cid = item["candidate_id"]
        score = item["score"]
        reason = generate_reasoning(item["candidate"], score, rank)
        rows.append({
            "candidate_id": cid,
            "rank": rank,
            "score": round(score, 4),
            "reasoning": reason,
            "_signals": item["candidate"].get("redrob_signals", {})
        })
        
    rows = make_unique_reasonings(rows)
    # Remove temporary _signals key
    for r in rows:
        if "_signals" in r:
            del r["_signals"]
            
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False, columns=["candidate_id", "rank", "score", "reasoning"])
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"Ranking completed in {duration:.2f} seconds. Output saved to {args.out}")

if __name__ == "__main__":
    main()
