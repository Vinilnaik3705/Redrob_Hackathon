import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
try:
    import faiss
except ImportError:
    class NumPyFaissFlatIP:
        def __init__(self, dimension):
            self.dimension = dimension
            self.embeddings = None
        def add(self, embeddings):
            self.embeddings = embeddings
        def search(self, query_vector, k):
            similarities = np.dot(self.embeddings, query_vector.T).flatten()
            top_k_indices = np.argsort(-similarities)[:k]
            top_k_distances = similarities[top_k_indices]
            return np.array([top_k_distances]), np.array([top_k_indices])

    class FaissFallback:
        IndexFlatIP = NumPyFaissFlatIP
        @staticmethod
        def normalize_L2(embeddings):
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-10, norms)
            np.divide(embeddings, norms, out=embeddings)

    faiss = FaissFallback()
import torch
import pickle
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from datetime import datetime
from sentence_transformers import SentenceTransformer
from scorer import (
    check_honeypots, check_disqualifications, clean_text,
    score_career_retrieval, score_production_deployment,
    score_behavioral_availability, score_skill_match,
    score_rules_context, score_yoe, calculate_penalties,
    is_consulting, JD_REQUIRED_SKILLS, score_assessment,
    score_career_trajectory
)

def reciprocal_rank_fusion(rank_lists, k=60):
    """
    rank_lists: list of ranked candidate index lists (best first)
    Returns: fused ranking as list of (index, fused_score) sorted descending
    """
    scores = {}
    for ranks in rank_lists:
        for position, idx in enumerate(ranks):
            scores[idx] = scores.get(idx, 0) + 1.0 / (k + position + 1)
    return sorted(scores.items(), key=lambda x: -x[1])

JD_QUERY = (
    "Senior AI Engineer with production experience in embeddings-based retrieval systems "
    "(sentence-transformers, BGE, E5, OpenAI embeddings), vector databases and hybrid search "
    "(Pinecone, Weaviate, Qdrant, Milvus, FAISS, Elasticsearch, OpenSearch), Python, "
    "ranking evaluation frameworks (NDCG, MRR, MAP, A/B testing, offline-to-online correlation). "
    "LLM fine-tuning (LoRA, QLoRA, PEFT), learning-to-rank. "
    "Startup product company experience. Located in India (Noida, Pune, Hyderabad, Bangalore, Mumbai, Delhi NCR)."
)

def calculate_rule_strength(candidate, semantic_score=None):
    """
    A pure rule-based 'ideal candidate' score, independent of any 
    retrieval/reranking model. Used as a floor/anchor signal.
    """
    p = candidate.get('profile', {})
    sig = candidate.get('redrob_signals', {})
    career = candidate.get('career_history', [])
    career_text = ' '.join(j.get('description','')+' '+j.get('title','') for j in career).lower()
    
    yoe = p.get('years_of_experience', 0)
    notice = sig.get('notice_period_days', 180)
    ret_count = sum(1 for kw in ['ranking','retrieval','recommendation','search','embedding','vector'] if kw in career_text)
    prod_count = sum(1 for kw in ['production','deployed','shipped','scale','serving'] if kw in career_text)
    
    strength = 0
    if 6 <= yoe <= 8: strength += 3
    if ret_count >= 3: strength += 3
    if prod_count >= 2: strength += 3
    if notice <= 15: strength += 2
    
    return strength

def normalize_dimensions(candidate_data):
    """Per-dimension min-max normalization across the scored candidate pool.

    After FAISS pre-filtering, each scoring dimension clusters in a narrow
    sub-range (e.g. semantic similarity 0.7-0.9 for all top-3000). Without
    normalization, the configured weight 0.30 only contributes ~0.06 of actual
    spread instead of 0.30. Normalizing each dimension to [0,1] across the pool
    ensures every weight reflects its intended discriminative contribution,
    producing natural score spread from real profile differences.
    """
    dims = ['semantic', 'retrieval', 'production', 'skill', 'rules', 'assessment', 'trajectory']
    for dim in dims:
        values = [cd[dim] for cd in candidate_data]
        dmin = min(values)
        dmax = max(values)
        if dmax > dmin:
            for cd in candidate_data:
                cd[f'{dim}_norm'] = (cd[dim] - dmin) / (dmax - dmin)
        else:
            for cd in candidate_data:
                cd[f'{dim}_norm'] = 0.5
    return candidate_data

def generate_reasoning(candidate, score, rank):
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])
    
    yoe = profile.get("years_of_experience", 0.0)
    title = profile.get("current_title", "Engineer")
    notice = signals.get("notice_period_days", 180)
    
    companies = []
    seen = set()
    for job in reversed(career):
        comp = job.get("company", "").strip()
        if comp and comp not in seen:
            companies.append(comp)
            seen.add(comp)
    
    if len(companies) > 3:
        companies_show = [companies[0], companies[-2], companies[-1]]
    else:
        companies_show = companies
    
    if companies_show:
        career_path = " > ".join(companies_show)
    else:
        career_path = "No previous company listed"
        
    location_str = profile.get("location", "India").split(",")[0].strip()
    if not location_str:
        location_str = "India"

    skills_list = [s.get("name", "") for s in candidate.get("skills", [])]
    matched = [s for s in skills_list if s.lower() in JD_REQUIRED_SKILLS]
    if matched:
        skills_str = ", ".join(matched[:3])
    else:
        skills_str = ", ".join(skills_list[:3]) if skills_list else "general ML skills"
        
    career_text = " ".join([j.get("description", "").lower() + " " + j.get("title", "").lower() for j in career])
    skills_lower = [s.lower() for s in skills_list]
    combined_context = (career_text + " " + " ".join(skills_lower)).lower()
    
    gaps = []
    if not any(kw in combined_context for kw in ["ranking", "rerank", "ndcg", "mrr", "map", "eval"]):
        gaps.append("ranking/eval")
    if not any(kw in combined_context for kw in ["faiss", "pinecone", "weaviate", "qdrant", "milvus", "vector"]):
        gaps.append("vector DBs")
    if not any(kw in combined_context for kw in ["sentence-transformers", "bge", "e5", "embedding"]):
        gaps.append("embeddings")
    if not any(kw in combined_context for kw in ["lora", "qlora", "peft", "llm", "fine-tuning"]):
        gaps.append("LLM fine-tuning")
    if not any(kw in combined_context for kw in ["production", "deployed", "shipped", "serving", "inference"]):
        gaps.append("production depth")
        
    gaps_str = f" Gaps: {', '.join(gaps)}." if gaps else ""

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
            availability = f"active {active_days_ago}d ago"
        else:
            availability = f"inactive {active_days_ago}d"
    else:
        last_days = signals.get("last_active_days_ago")
        if last_days is not None:
            if last_days <= 30:
                availability = "recently active"
            elif last_days <= 90:
                availability = f"active {last_days}d ago"
            else:
                availability = f"inactive {last_days}d"
        else:
            availability = "activity unknown"

    completeness = signals.get("profile_completeness_score", 50)
    has_verified = signals.get("verified_email", False) or signals.get("verified_phone", False) or signals.get("linkedin_connected", False)
    if completeness >= 80 and has_verified:
        confidence = "HIGH"
    elif completeness >= 50:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    risks = []
    tenures = [j.get("duration_months", 24) for j in career[:3] if j.get("duration_months") is not None]
    if tenures:
        avg_tenure = sum(tenures) / len(tenures)
        if avg_tenure < 15:
            risks.append("job hopping")
    offer_rate = signals.get("offer_acceptance_rate", -1)
    if 0 <= offer_rate < 0.4:
        risks.append("low offer acceptance")
    if notice > 90:
        risks.append("long notice")
    
    risk_level = "LOW"
    if len(risks) >= 2:
        risk_level = "HIGH"
    elif len(risks) == 1:
        risk_level = "MEDIUM"
        
    risk_str = f" [Risk: {risk_level}]" if risks else ""

    reasoning = (
        f"{title} ({career_path}), {yoe:.1f}yrs; strong in {skills_str}. "
        f"{availability.capitalize()}, {notice}d notice, {location_str}.{gaps_str} "
        f"[Confidence: {confidence}]{risk_str}"
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

def get_candidate_text(c):
    profile = c.get("profile", {})
    headline = profile.get("headline", "")
    summary = profile.get("summary", "")
    career_texts = []
    for job in c.get("career_history", []):
        career_texts.append(f"{job.get('title', '')} at {job.get('company', '')}: {job.get('description', '')}")
    skills_str = " ".join([s.get("name", "") for s in c.get("skills", [])])
    return clean_text(f"{headline} {summary} {' '.join(career_texts)} {skills_str}")

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
            for line_idx, line in enumerate(f):
                if line.strip():
                    try:
                        candidates.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipped invalid JSON line {line_idx+1}: {e}")
                    
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
            
            faiss.normalize_L2(survivor_embeddings)
            faiss.normalize_L2(jd_vector)
            
            semantic_scores = np.dot(survivor_embeddings, jd_vector.flatten())
            
            bm25_path = os.path.join(base_dir, "bm25_index.pkl")
            if not os.path.exists(bm25_path):
                print(f"Error: {bm25_path} not found. Please run precompute.py first.")
                sys.exit(1)
            with open(bm25_path, "rb") as f:
                bm25_data = pickle.load(f)
            
            jd_tokens = jd_query.lower().split()
            bm25_scores = bm25_data["bm25"].get_scores(jd_tokens)
            
            dense_rank = sorted(range(len(survivor_indices)), key=lambda i: -semantic_scores[i])
            bm25_rank = sorted(range(len(survivor_indices)), key=lambda i: -bm25_scores[survivor_indices[i]])
            
            fused = reciprocal_rank_fusion([dense_rank, bm25_rank])
            shortlist_surv_indices = [idx for idx, _ in fused[:2000]]
            print(f"Shortlisted top {len(shortlist_surv_indices)} candidates using RRF fusion.")
            
            candidate_data = []
            for surv_idx in shortlist_surv_indices:
                orig_idx = survivor_indices[surv_idx]
                semantic_score = float(semantic_scores[surv_idx])
                
                feat = pre_features[orig_idx]
                
                if len(feat) > 8:
                    assessment_val = float(feat[7])
                    trajectory_val = float(feat[8])
                else:
                    assessment_val = score_assessment(candidates[orig_idx])
                    trajectory_val = score_career_trajectory(candidates[orig_idx])
                
                candidate_data.append({
                    'orig_idx': orig_idx,
                    'semantic': semantic_score,
                    'retrieval': float(feat[0]),
                    'production': float(feat[1]),
                    'skill': float(feat[2]),
                    'behavioral': float(feat[3]),
                    'rules': float(feat[4]),
                    'yoe': float(feat[5]),
                    'penalties': float(feat[6]),
                    'assessment': assessment_val,
                    'trajectory': trajectory_val,
                })
            
            candidate_data = normalize_dimensions(candidate_data)
            
            for cd in candidate_data:
                technical_score = (
                    0.28 * cd['semantic_norm'] +
                    0.22 * cd['retrieval_norm'] +
                    0.18 * cd['production_norm'] +
                    0.12 * cd['skill_norm'] +
                    0.08 * cd['trajectory_norm'] +
                    0.07 * cd['assessment_norm'] +
                    0.05 * cd['rules_norm']
                )
                
                behavioral_mult = 0.25 + 0.75 * cd['behavioral']
                
                final_score = technical_score * behavioral_mult - cd['penalties']
                final_score = max(0.0, final_score)
                
                if cd['retrieval'] == 0.0:
                    base = cd['semantic_norm'] * 0.25
                    final_score = max(0.0, base)
                elif cd['retrieval'] < 0.3:
                    multiplier = 0.5 + (cd['retrieval'] / 0.3) * 0.3
                    final_score = final_score * multiplier
                
                scored_candidates.append({
                    "candidate_id": candidates[cd['orig_idx']]["candidate_id"],
                    "score": final_score,
                    "candidate": candidates[cd['orig_idx']]
                })
                
            scored_candidates.sort(key=lambda x: -x["score"])
            top_500 = scored_candidates[:500]
            
            pairs = [(jd_query, get_candidate_text(item["candidate"])) for item in top_500]
            
            print("Reranking top 500 candidates with Cross-Encoder on CPU...")
            cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')
            cross_scores = cross_encoder.predict(pairs)
            
            s_min = min(cross_scores)
            s_max = max(cross_scores)
            if s_max > s_min:
                normalized_cross = [(s - s_min) / (s_max - s_min) for s in cross_scores]
            else:
                normalized_cross = [0.5] * len(cross_scores)
                
            for i, item in enumerate(top_500):
                blended_score = 0.3 * normalized_cross[i] + 0.7 * item["score"]
                item["score"] = blended_score
            
            scored_ids = {x["candidate_id"] for x in scored_candidates}
            for idx in range(len(candidates)):
                cid = candidates[idx]["candidate_id"]
                if cid not in scored_ids:
                    scored_candidates.append({
                        "candidate_id": cid,
                        "score": 0.0,
                        "candidate": candidates[idx]
                    })
                    
        scored_candidates = [x for x in scored_candidates if x["score"] > 0.0]
        scored_candidates.sort(key=lambda x: (
            -round(x["score"], 4), 
            -calculate_rule_strength(x["candidate"], None), 
            x["candidate_id"]
        ))
        top_items = scored_candidates[:min(args.limit, len(scored_candidates))]
    else:
        print("Computing features and embeddings dynamically (fallback)...")
        model = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cuda' if torch.cuda.is_available() else 'cpu')
        
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
            texts = []
            for c in survivors:
                texts.append(get_candidate_text(c))
                
            print("Embedding survivors...")
            surv_embeddings = model.encode(texts, batch_size=256, convert_to_numpy=True).astype(np.float32)
            
            print("Embedding job description...")
            jd_vector = model.encode([jd_query])[0].astype(np.float32)
            jd_vector = jd_vector.reshape(1, -1)
            
            faiss.normalize_L2(surv_embeddings)
            faiss.normalize_L2(jd_vector)
            
            semantic_scores = np.dot(surv_embeddings, jd_vector.flatten())
            
            print("Building BM25 index dynamically (fallback)...")
            tokenized_corpus = [text.lower().split() for text in texts]
            bm25 = BM25Okapi(tokenized_corpus)
            
            jd_tokens = jd_query.lower().split()
            bm25_scores = bm25.get_scores(jd_tokens)
            
            dense_rank = sorted(range(len(survivors)), key=lambda i: -semantic_scores[i])
            bm25_rank = sorted(range(len(survivors)), key=lambda i: -bm25_scores[i])
            
            fused = reciprocal_rank_fusion([dense_rank, bm25_rank])
            shortlist_surv_indices = [idx for idx, _ in fused[:2000]]
            print(f"Shortlisted top {len(shortlist_surv_indices)} candidates using RRF fusion.")
            
            candidate_data = []
            for surv_idx in shortlist_surv_indices:
                c = survivors[surv_idx]
                semantic_score = float(semantic_scores[surv_idx])
                
                retrieval_score = score_career_retrieval(c)
                production_score = score_production_deployment(c)
                skill_score = score_skill_match(c)
                behavioral_score = score_behavioral_availability(c)
                rules_score_val = score_rules_context(c)
                penalties_total = calculate_penalties(c)
                assessment_val = score_assessment(c)
                trajectory_val = score_career_trajectory(c)
                
                candidate_data.append({
                    'candidate': c,
                    'orig_idx': survivor_orig_indices[surv_idx],
                    'semantic': semantic_score,
                    'retrieval': retrieval_score,
                    'production': production_score,
                    'skill': skill_score,
                    'behavioral': behavioral_score,
                    'rules': rules_score_val,
                    'penalties': penalties_total,
                    'assessment': assessment_val,
                    'trajectory': trajectory_val,
                })
            
            candidate_data = normalize_dimensions(candidate_data)
            
            for cd in candidate_data:
                technical_score = (
                    0.28 * cd['semantic_norm'] +
                    0.22 * cd['retrieval_norm'] +
                    0.18 * cd['production_norm'] +
                    0.12 * cd['skill_norm'] +
                    0.08 * cd['trajectory_norm'] +
                    0.07 * cd['assessment_norm'] +
                    0.05 * cd['rules_norm']
                )
                
                behavioral_mult = 0.25 + 0.75 * cd['behavioral']
                
                final_score = technical_score * behavioral_mult - cd['penalties']
                final_score = max(0.0, final_score)
                
                if cd['retrieval'] == 0.0:
                    base = cd['semantic_norm'] * 0.25
                    final_score = max(0.0, base)
                elif cd['retrieval'] < 0.3:
                    multiplier = 0.5 + (cd['retrieval'] / 0.3) * 0.3
                    final_score = final_score * multiplier
                
                scored_candidates.append({
                    "candidate_id": cd['candidate']["candidate_id"],
                    "score": final_score,
                    "candidate": cd['candidate']
                })
                
            scored_candidates.sort(key=lambda x: -x["score"])
            top_500 = scored_candidates[:500]
            
            pairs = [(jd_query, get_candidate_text(item["candidate"])) for item in top_500]
            
            print("Reranking top 500 candidates with Cross-Encoder on CPU...")
            cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')
            cross_scores = cross_encoder.predict(pairs)
            
            s_min = min(cross_scores)
            s_max = max(cross_scores)
            if s_max > s_min:
                normalized_cross = [(s - s_min) / (s_max - s_min) for s in cross_scores]
            else:
                normalized_cross = [0.5] * len(cross_scores)
                
            for i, item in enumerate(top_500):
                blended_score = 0.3 * normalized_cross[i] + 0.7 * item["score"]
                item["score"] = blended_score
            
            scored_ids = {x["candidate_id"] for x in scored_candidates}
            for c in candidates:
                if c["candidate_id"] not in scored_ids:
                    scored_candidates.append({
                        "candidate_id": c["candidate_id"],
                        "score": 0.0,
                        "candidate": c
                    })
                    
        scored_candidates = [x for x in scored_candidates if x["score"] > 0.0]
        scored_candidates.sort(key=lambda x: (
            -round(x["score"], 4), 
            -calculate_rule_strength(x["candidate"], None), 
            x["candidate_id"]
        ))
        top_items = scored_candidates[:min(args.limit, len(scored_candidates))]
            
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
