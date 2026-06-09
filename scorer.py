import re
from datetime import datetime

# JD Skills Database
JD_SKILLS = {
    "embeddings", "vector databases", "python", "ndcg", "mrr", "map", 
    "fine-tuning", "learning-to-rank", "open-source", "faiss", 
    "pinecone", "qdrant", "weaviate", "milvus", "sentence-transformers", 
    "bge", "e5", "lora", "qlora", "rag", "hybrid search", 
    "elasticsearch", "opensearch", "learning to rank", "peft", "xgboost"
}

# Fictional & Real Startup Founded Years
FOUNDED_YEARS = {
    "Krutrim": 2023,
    "Sarvam AI": 2023,
    "CRED": 2018,
    "Razorpay": 2014,
    "Swiggy": 2014,
    "Zomato": 2008,
    "Flipkart": 2007,
    "Ola": 2010,
    "PhonePe": 2015,
    "Paytm": 2010,
    "Aganitha": 2018,
    "Genpact AI": 2023,
    "Haptik": 2013,
    "Meesho": 2015,
    "Niramai": 2016,
    "Observe.AI": 2017,
    "Rephrase.ai": 2019,
    "Saarthi.ai": 2017,
    "Yellow.ai": 2016,
    "Zoho": 1996,
    "upGrad": 2015,
    "Wysa": 2015,
    "Verloop.io": 2015,
}

CONSULTING_COMPANIES = {
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture", 
    "cognizant", "capgemini", "tech mahindra", "l&t infotech", "lti", 
    "mindtree", "hcl", "hcltech", "hcl technologies", "deloitte", "kpmg", 
    "ey", "pwc", "pwc limited", "ernst & young", "pricewaterhousecoopers", "capgemini"
}

def clean_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text).strip()

def calculate_skill_match_score(resume_skills: list) -> float:
    """Score based on overlap between JD-required skills and resume skills."""
    if not resume_skills:
        return 0.0

    resume_skills_set = set(s.lower().strip() for s in resume_skills if s and str(s).strip())
    exact_matched = JD_SKILLS.intersection(resume_skills_set)
    # Use 6.0 as denominator since no candidate has all 26 skills, capped at 1.0
    exact_score = len(exact_matched) / 6.0

    unmatched_jd = JD_SKILLS - exact_matched
    partial_score = 0.0
    for jd_skill in unmatched_jd:
        for r_skill in resume_skills_set:
            if jd_skill in r_skill or r_skill in jd_skill:
                partial_score += 0.5 / 6.0
                break

    return min(exact_score + partial_score, 1.0)

def is_consulting(company_name: str, industry: str) -> bool:
    if not company_name:
        return False
    company_lower = company_name.lower().strip()
    if any(c in company_lower for c in CONSULTING_COMPANIES):
        return True
    if industry and industry.lower() in ["it services", "it services and it consulting", "consulting"]:
        return True
    return False

def check_career_for_retrieval(career_history: list) -> bool:
    retrieval_keywords = {"search", "ranking", "retrieval", "recommendation", "matching", "relevance", "similarity", "vector", "embedding", "index", "recall", "precision", "reranking", "candidate generation", "rag", "dense retrieval", "hybrid search"}
    for job in career_history:
        desc = job.get("description", "").lower()
        title = job.get("title", "").lower()
        if any(kw in desc or kw in title for kw in retrieval_keywords):
            return True
    return False

def check_career_for_production(career_history: list) -> float:
    production_keywords = ["production", "deployed", "shipped", "launched", "real users", "scale", "serving", "inference", "a/b test", "experiment", "online eval", "optimiz", "architecture", "pipelines"]
    count = 0
    for job in career_history:
        desc = job.get("description", "").lower()
        title = job.get("title", "").lower()
        count += sum(1 for kw in production_keywords if kw in desc or kw in title)
    if count >= 3:
        return 1.0
    return count / 3.0

def is_title_chaser(career_history: list) -> bool:
    if len(career_history) < 2:
        return False
    short_tenures = 0
    for job in career_history[:3]:
        months = job.get("duration_months")
        if months is not None and months < 18:
            short_tenures += 1
    return short_tenures >= 2

def is_research_only(career_history: list) -> bool:
    if not career_history:
        return False
    research_titles = ["research scientist", "research engineer", "research intern", "phd student", "postdoc", "research fellow", "ml researcher", "ai researcher"]
    all_research = True
    has_production_signal = False
    production_keywords = ["production", "deployed", "shipped", "launched", "real users", "serving", "inference"]
    
    for job in career_history:
        title = job.get("title", "").lower()
        desc = job.get("description", "").lower()
        if not any(rt in title for rt in research_titles):
            all_research = False
        if any(pk in desc or pk in title for pk in production_keywords):
            has_production_signal = True
            
    return all_research and not has_production_signal

def is_langchain_only(candidate: dict) -> bool:
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    signup_date_str = signals.get("signup_date", "")
    if not signup_date_str:
        return False
    try:
        today = datetime(2026, 6, 7)
        signup_date = datetime.strptime(signup_date_str, "%Y-%m-%d")
        if (today - signup_date).days < 365:
            framework_skills = {"langchain", "llamaindex", "llama-index", "autogpt", "langflow"}
            skill_names = set(s.get("name", "").lower() for s in skills)
            has_framework = any(f in skill_names for f in framework_skills)
            fundamental_skills = {"machine learning", "deep learning", "nlp", "information retrieval", "retrieval", "computer vision", "statistics", "math"}
            has_fundamental = any(fun in skill_names for fun in fundamental_skills)
            if has_framework and not has_fundamental:
                return True
    except Exception:
        pass
    return False

def is_closed_source_only(candidate: dict) -> bool:
    profile = candidate.get("profile", {})
    yoe = profile.get("years_of_experience", 0.0)
    signals = candidate.get("redrob_signals", {})
    github = signals.get("github_activity_score", -1)
    if yoe >= 5.0 and github == -1:
        return True
    return False

# ============ NEW HELPER FUNCTIONS FOR 3-STAGE SCORER ============

def in_india(candidate: dict) -> bool:
    profile = candidate.get("profile", {})
    loc = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    return country == "india" or "india" in loc

def inactive_over_1_year(candidate: dict) -> bool:
    signals = candidate.get("redrob_signals", {})
    last_active = signals.get("last_active_date", "")
    if not last_active:
        return False
    try:
        today = datetime(2026, 6, 7)
        active_date = datetime.strptime(last_active, "%Y-%m-%d")
        days = (today - active_date).days
        return days > 365
    except Exception:
        return False

def is_pure_consulting(candidate: dict) -> bool:
    career_history = candidate.get("career_history", [])
    if not career_history:
        return False
    return all(is_consulting(job.get("company", ""), job.get("industry", "")) for job in career_history)

DISQUALIFIED_TITLES = [
    'business analyst', 'hr manager', 'human resources',
    'customer support', 'customer success', 'full stack developer',
    'cloud engineer', 'devops', 'project manager', 'program manager',
    'marketing manager', 'sales', 'mechanical engineer',
    'civil engineer', 'finance', 'accountant'
]

def is_unrelated_role(candidate: dict) -> bool:
    import re
    profile = candidate.get("profile", {})
    title = profile.get("current_title", "").lower()
    
    # Check disqualified titles
    for dt in DISQUALIFIED_TITLES:
        if dt == 'sales':
            if re.search(r'\bsales\b', title, re.IGNORECASE) and 'salesforce' not in title:
                return True
        elif dt in title:
            return True
    return False

FICTIONAL_COMPANIES = {
    'hooli', 'pied piper', 'globex', 'initech', 'dunder mifflin',
    'wayne enterprises', 'stark industries', 'acme corp',
    'umbrella corporation', 'cyberdyne', 'weyland', 'buy n large',
    'piedpiper', 'dundermifflin', 'wayneenterprises', 'starkindustries',
    'umbrellacorporation', 'cyberdyne systems', 'weyland-yutani', 'weylandyutani'
}

def is_honeypot(candidate: dict) -> bool:
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    title = profile.get("current_title", "")
    career_history = candidate.get("career_history", [])
    yoe = profile.get("years_of_experience", 0.0)
    
    # 1. Junior title with senior YOE contradictory flag
    if 'junior' in title.lower() and yoe > 5.0:
        return True
        
    # 2. Fictional company check
    for job in career_history:
        company = job.get('company', '').lower()
        if any(fc in company for fc in FICTIONAL_COMPANIES):
            return True
            
    # 3. Timeline check
    for job in career_history:
        comp = job.get("company", "")
        start = job.get("start_date")
        if comp in FOUNDED_YEARS and start:
            try:
                start_yr = int(start.split("-")[0])
                if start_yr < FOUNDED_YEARS[comp]:
                    return True
            except Exception:
                pass
                
    expert_zero_end = [s for s in skills if s.get("proficiency") == "expert" and s.get("endorsements", 0) == 0]
    if len(expert_zero_end) >= 8:
        return True
    expert_zero_dur = [s for s in skills if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0]
    if len(expert_zero_dur) >= 8:
        return True
        
    if "marketing manager" in title.lower():
        ml_keywords = {"ml", "machine learning", "deep learning", "nlp", "computer vision", "cv", "tensorflow", "pytorch", "embeddings", "vector", "ai", "retrieval", "search", "ranking"}
        has_ml = False
        for s in skills:
            skill_name_lower = s["name"].lower()
            words = set(w.strip() for w in skill_name_lower.replace("-", " ").replace("/", " ").split())
            if words.intersection(ml_keywords) or skill_name_lower in ml_keywords:
                has_ml = True
                break
        if has_ml:
            return True
            
    return False

def score_career_retrieval(candidate: dict) -> float:
    career_history = candidate.get("career_history", [])
    retrieval_keywords = {"search", "ranking", "retrieval", "recommendation", "matching", "relevance", "similarity", "vector", "embedding", "index", "recall", "precision", "reranking", "candidate generation", "rag", "dense retrieval", "hybrid search", "weaviate", "pinecone", "qdrant", "milvus", "elasticsearch", "opensearch", "faiss"}
    matches = 0
    for job in career_history:
        desc = job.get("description", "").lower()
        title = job.get("title", "").lower()
        matches += sum(1 for kw in retrieval_keywords if kw in desc or kw in title)
    return min(1.0, matches / 3.0)

def score_career_production(candidate: dict) -> float:
    career_history = candidate.get("career_history", [])
    production_keywords = ["production", "deployed", "shipped", "launched", "real users", "scale", "serving", "inference", "a/b test", "experiment", "online eval", "optimiz", "architecture", "pipelines"]
    matches = 0
    for job in career_history:
        desc = job.get("description", "").lower()
        title = job.get("title", "").lower()
        matches += sum(1 for kw in production_keywords if kw in desc or kw in title)
    return min(1.0, matches / 3.0)

def calculate_yoe_score(yoe: float) -> float:
    if 6.0 <= yoe <= 8.0:
        return 1.0
    elif 4.0 <= yoe < 6.0:
        return 0.75
    elif 8.0 < yoe <= 10.0:
        return 0.65
    elif yoe > 10.0:
        return 0.4
    else:
        return 0.2

def calculate_behavioral_score(candidate: dict) -> float:
    signals = candidate.get("redrob_signals", {})
    last_active = signals.get("last_active_date", "")
    
    active_score = 0.0
    if last_active:
        try:
            today = datetime(2026, 6, 7)
            active_date = datetime.strptime(last_active, "%Y-%m-%d")
            days = (today - active_date).days
            if days <= 7:
                active_score = 1.0
            elif days <= 30:
                active_score = 0.7
            elif days <= 90:
                active_score = 0.4
            elif days <= 180:
                active_score = 0.1
        except Exception:
            pass
            
    open_to_work = 0.3 if signals.get("open_to_work_flag", False) else 0.0
    response_rate = float(signals.get("recruiter_response_rate", 0.0))
    
    notice_days = signals.get("notice_period_days", 180)
    notice_score = 0.0
    if notice_days <= 30:
        notice_score = 1.0
    elif notice_days <= 60:
        notice_score = 0.6
    elif notice_days <= 90:
        notice_score = 0.3
        
    interview_completion = float(signals.get("interview_completion_rate", 0.0))
    
    raw_behavioral = active_score + open_to_work + response_rate + notice_score + interview_completion
    return raw_behavioral / 4.3

def calculate_skills_corroborated(candidate: dict) -> float:
    skills = candidate.get("skills", [])
    if not skills:
        return 0.0
    career_history = candidate.get("career_history", [])
    career_desc = " ".join([j.get("description", "").lower() + " " + j.get("title", "").lower() for j in career_history])
    corroborated = 0
    count = 0
    for s in skills:
        s_name = s.get("name", "").lower()
        if s_name in JD_SKILLS:
            count += 1
            if s_name in career_desc:
                corroborated += 1
    if count == 0:
        return 0.5
    return corroborated / count

def calculate_rules(candidate: dict) -> float:
    profile = candidate.get("profile", {})
    career_history = candidate.get("career_history", [])
    signals = candidate.get("redrob_signals", {})
    
    score = 0.0
    
    # 1. Product company preference
    has_product_role = False
    for job in career_history:
        comp = job.get("company", "")
        ind = job.get("industry", "")
        if comp and not is_consulting(comp, ind):
            has_product_role = True
            break
    if has_product_role:
        score += 0.8
        
    # 2. Location preference
    loc = profile.get("location", "").lower()
    preferred_cities = ["pune", "noida", "hyderabad", "mumbai", "delhi", "ncr", "bangalore", "bengaluru"]
    if any(city in loc for city in preferred_cities):
        score += 0.6
            
    # 3. Willing to relocate
    if signals.get("willing_to_relocate", False):
        score += 0.5
        
    # Nice-to-haves: HR-tech/marketplace background (+0.1)
    hrtech_keywords = {"hr tech", "recruiting", "talent", "marketplace", "staffing", "hr-tech"}
    for job in career_history:
        ind = job.get("industry", "").lower()
        desc = job.get("description", "").lower()
        if any(hk in ind or hk in desc for hk in hrtech_keywords):
            score += 0.1
            break
            
    # Nice-to-haves: GitHub activity score
    github_score = signals.get("github_activity_score", -1)
    if github_score > 0:
        score += min(0.2, github_score / 500.0)
        
    max_possible = 2.2 # 0.8 + 0.6 + 0.5 + 0.1 + 0.2
    return min(1.0, max(0.0, score / max_possible))

# Penalties helper calculations
def title_chaser_penalty(candidate: dict) -> float:
    return 0.12 if is_title_chaser(candidate.get("career_history", [])) else 0.0

def langchain_only_penalty(candidate: dict) -> float:
    return 0.10 if is_langchain_only(candidate) else 0.0

def research_only_penalty(candidate: dict) -> float:
    return 0.15 if is_research_only(candidate.get("career_history", [])) else 0.0

def notice_period_penalty(candidate: dict) -> float:
    signals = candidate.get("redrob_signals", {})
    notice_days = signals.get("notice_period_days", 180)
    if notice_days > 90:
        return 0.12
    elif notice_days > 60:
        return 0.08
    elif notice_days > 30:
        return 0.04
    return 0.0

# ============ MAIN ENTRANCE SCORER ============

def score_candidate(candidate: dict, semantic_score: float) -> float:
    # ============ STAGE 1: HARD KNOCKOUTS ============
    profile = candidate.get("profile", {})
    yoe = profile.get("years_of_experience", 0.0)
    
    if is_honeypot(candidate):           return 0.0
    if is_unrelated_role(candidate):     return 0.0
    if is_pure_consulting(candidate):    return 0.0
    if yoe < 3.0:                        return 0.0
    if not in_india(candidate):          return 0.0
    if inactive_over_1_year(candidate):  return 0.0
    
    # ============ STAGE 2: CORE EVIDENCE ============
    retrieval_evidence = score_career_retrieval(candidate)
    production_evidence = score_career_production(candidate)
    
    # Hard cap if zero retrieval evidence in entire career
    if retrieval_evidence == 0.0:
        return float(min(0.3, semantic_score * 0.3))
        
    core_score = 0.5 * retrieval_evidence + 0.5 * production_evidence
    
    # ============ STAGE 3: FINE RANKING ============
    yoe_score = calculate_yoe_score(yoe)
    behavioral = calculate_behavioral_score(candidate)
    skills_corr = calculate_skills_corroborated(candidate)
    
    fine_score = 0.25 * yoe_score + 0.50 * behavioral + 0.25 * skills_corr
    
    # ============ FINAL COMBINATION ============
    # Base formula
    base = (0.35 * semantic_score +   # JD semantic match
            0.40 * core_score +        # career evidence - dominant
            0.15 * fine_score +        # differentiation signals
            0.10 * calculate_rules(candidate))  # location, company type
            
    # Apply JD disqualifier penalties
    base -= title_chaser_penalty(candidate)
    base -= langchain_only_penalty(candidate)
    base -= research_only_penalty(candidate)
    base -= notice_period_penalty(candidate)
    
    # Additional penalty for Junior Research Engineers
    title = profile.get("current_title", "").lower()
    if 'research' in title and yoe < 5.0:
        base -= 0.25
        
    return float(max(0.0, base))

# Alias for backward compatibility with precompute.py
calculate_rules_score = calculate_rules
