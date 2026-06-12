import re
from datetime import datetime

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Fictional & trap companies (Honeypot checking)
FICTIONAL_COMPANIES = {
    "hooli", "pied piper", "globex", "initech", "dunder mifflin",
    "wayne enterprises", "stark industries", "acme corp", "umbrella corporation",
    "cyberdyne", "piedpiper", "dundermifflin", "wayneenterprises", "starkindustries"
}

# Real founded years to catch timeline impossibility traps
KNOWN_FOUNDED_YEARS = {
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

# Disqualified exact titles to eliminate unrelated applications
DISQUALIFIED_EXACT_TITLES = {
    "hr manager", "human resources", "marketing manager", "sales manager",
    "mechanical engineer", "civil engineer", "graphic designer",
    "project manager", "program manager", "business analyst",
    "customer support", "customer success", "accountant", "finance manager"
}

# Target cities preferred by the hiring managers
PREFERRED_CITIES = {
    "pune", "noida", "hyderabad", "mumbai", "delhi", "ncr", "bangalore", "bengaluru"
}

# Large consulting/IT services companies
CONSULTING_COMPANIES = {
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture", 
    "cognizant", "capgemini", "tech mahindra", "l&t infotech", "lti", 
    "mindtree", "hcl", "hcltech", "hcl technologies", "deloitte", "kpmg", 
    "ey", "pwc", "pwc limited", "ernst & young", "pricewaterhousecoopers", "ltimindtree"
}

# JD Required & Preferred skills for matching
JD_REQUIRED_SKILLS = {
    "embeddings", "vector databases", "python", "ndcg", "mrr", "map",
    "fine-tuning", "learning-to-rank", "faiss", "pinecone", "qdrant",
    "weaviate", "milvus", "sentence-transformers", "bge", "e5",
    "lora", "qlora", "rag", "hybrid search", "elasticsearch",
    "opensearch", "peft", "xgboost"
}


def clean_text(text: str) -> str:
    """Clean and normalize whitespaces in text."""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text).strip()

def check_honeypots(c: dict) -> bool:
    """Evaluate 7 honeypot rules to trap fake/impossible candidates."""
    profile = c.get("profile", {})
    title = profile.get("current_title", "").lower()
    yoe = profile.get("years_of_experience", 0.0)
    career_history = c.get("career_history", [])
    skills = c.get("skills", [])
    signals = c.get("redrob_signals", {})
    
    # Rule 1 - Timeline Impossibility
    for job in career_history:
        comp = job.get("company", "")
        start = job.get("start_date")
        if comp in KNOWN_FOUNDED_YEARS and start:
            try:
                start_yr = int(start.split("-")[0])
                if start_yr < KNOWN_FOUNDED_YEARS[comp]:
                    return True
            except:
                pass
                
    # Rule 2 - Junior Title + Senior YOE
    if "junior" in title and yoe >= 5.0:
        return True
        
    # Rule 3 - Fictional Company
    for job in career_history:
        comp_lower = job.get("company", "").lower().strip()
        if any(fc in comp_lower for fc in FICTIONAL_COMPANIES):
            return True
            
    # Rule 4 - Expert Skills with Zero Endorsements / Duration
    expert_zero_end = sum(1 for s in skills if s.get("proficiency") == "expert" and s.get("endorsements", 0) == 0)
    expert_zero_dur = sum(1 for s in skills if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0)
    if expert_zero_end >= 8 or expert_zero_dur >= 8:
        return True
        
    # Rule 5 - Marketing/HR Title with ML Skills
    if title == "marketing manager" or title == "hr manager":
        ml_keywords = {"ml", "machine learning", "deep learning", "nlp", "computer vision", "cv", "tensorflow", "pytorch", "embeddings", "vector", "ai", "retrieval"}
        skill_names = set(s.get("name", "").lower() for s in skills)
        if any(kw in skill_names for kw in ml_keywords):
            return True
            
    # Rule 6 - Too-Perfect Profile
    flags = 0
    github_score = signals.get("github_activity_score", -1)
    if github_score > 950:
        flags += 1
    if signals.get("recruiter_response_rate") == 1.0:
        flags += 1
    if signals.get("interview_completion_rate") == 1.0:
        flags += 1
    if yoe in [5.0, 10.0, 15.0, 20.0]:
        assessments = signals.get("skill_assessments", [])
        if assessments and all(float(a.get("score", 0)) >= 98 for a in assessments):
            flags += 1
    if len(skills) > 40:
        flags += 1
    if flags >= 3:
        return True
        
    # Rule 7 - All Skill Assessments Perfect
    assessments = signals.get("skill_assessments", [])
    if len(assessments) >= 3 and all(float(a.get("score", 0)) >= 98 for a in assessments):
        return True
        
    return False

def check_disqualifications(c: dict) -> bool:
    """Verify core eligibility and qualification rules."""
    profile = c.get("profile", {})
    title = profile.get("current_title", "").lower().strip()
    country = profile.get("country", "").lower().strip()
    loc = profile.get("location", "").lower().strip()
    yoe = profile.get("years_of_experience", 0.0)
    signals = c.get("redrob_signals", {})
    last_active = signals.get("last_active_date", "")
    
    # Rule 8 - Unrelated Title
    if title in DISQUALIFIED_EXACT_TITLES:
        return True
    if re.search(r'\bsales\b', title) and "salesforce" not in title:
        return True
        
    # Rule 9 - Outside India
    if country != "india" and "india" not in loc:
        return True
        
    # Rule 10 - Inactive > 12 Months
    if last_active:
        try:
            today = datetime(2026, 6, 7)
            active_date = datetime.strptime(last_active, "%Y-%m-%d")
            if (today - active_date).days > 365:
                return True
        except:
            pass
            
    # Rule 11 - YOE Hard Minimum
    if yoe < 3.0:
        return True
        
    # Rule 12 - Pure Research Scientist without production experience
    if is_research_only(c.get("career_history", [])):
        return True
        
    return False

# ============================================================================
# SCORING DIMENSIONS
# ============================================================================

def score_career_retrieval(c: dict) -> float:
    """Measure the depth of retrieval experience in career history."""
    career_history = c.get("career_history", [])
    retrieval_keywords = {
        "search", "ranking", "retrieval", "recommendation", "matching", "relevance",
        "similarity", "vector", "embedding", "index", "recall", "precision",
        "reranking", "candidate generation", "rag", "dense retrieval", "hybrid search",
        "weaviate", "pinecone", "qdrant", "milvus", "elasticsearch", "opensearch", "faiss"
    }
    count = 0
    for job in career_history:
        desc = job.get("description", "").lower()
        title = job.get("title", "").lower()
        count += sum(1 for kw in retrieval_keywords if kw in desc or kw in title)
    return min(1.0, count / 4.0)

def score_production_deployment(c: dict) -> float:
    """Measure deployment and production engineering experience."""
    career_history = c.get("career_history", [])
    production_keywords = [
        "production", "deployed", "shipped", "launched", "real users", "scale",
        "serving", "inference", "a/b test", "experiment", "online eval",
        "optimiz", "architecture", "pipelines", "latency", "throughput"
    ]
    count = 0
    for job in career_history:
        desc = job.get("description", "").lower()
        title = job.get("title", "").lower()
        count += sum(1 for kw in production_keywords if kw in desc or kw in title)
    return min(1.0, count / 4.0)

def score_behavioral_availability(c: dict) -> float:
    """Evaluate candidate response rates, activity, and notice periods."""
    signals = c.get("redrob_signals", {})
    last_active = signals.get("last_active_date", "")
    
    last_active_score = 0.0
    if last_active:
        try:
            today = datetime(2026, 6, 7)
            active_date = datetime.strptime(last_active, "%Y-%m-%d")
            days = (today - active_date).days
            if days <= 7:
                last_active_score = 1.0
            elif days <= 30:
                last_active_score = 0.75
            elif days <= 90:
                last_active_score = 0.45
            elif days <= 180:
                last_active_score = 0.15
        except:
            pass
            
    open_to_work_score = 0.3 if signals.get("open_to_work_flag", False) else 0.0
    response_rate_score = float(signals.get("recruiter_response_rate", 0.0))
    
    notice_days = signals.get("notice_period_days", 180)
    notice_score = 0.0
    if notice_days <= 30:
        notice_score = 1.0
    elif notice_days <= 60:
        notice_score = 0.6
    elif notice_days <= 90:
        notice_score = 0.3
        
    interview_completion_score = float(signals.get("interview_completion_rate", 0.0))
    
    raw = last_active_score + open_to_work_score + response_rate_score + notice_score + interview_completion_score
    return raw / 4.3

def score_skill_match(c: dict) -> float:
    """Corroborate skills list against career descriptions and endorsements."""
    skills = c.get("skills", [])
    if not skills:
        return 0.0
        
    skill_names = {s.get("name", "").lower().strip() for s in skills}
    matched = JD_REQUIRED_SKILLS.intersection(skill_names)
    exact_score = len(matched) / 6.0
    exact_score = min(exact_score, 1.0)
    
    career_history = c.get("career_history", [])
    career_text = " ".join([j.get("description", "").lower() + " " + j.get("title", "").lower() for j in career_history])
    
    corroborated = sum(1 for s in matched if s in career_text)
    corroboration_ratio = corroborated / max(len(matched), 1)
    
    endorsed_skills = [s for s in skills if s.get("endorsements", 0) > 0 and s.get("name", "").lower() in JD_REQUIRED_SKILLS]
    endorsement_boost = min(0.2, len(endorsed_skills) / 10.0)
    
    return min(1.0, exact_score * corroboration_ratio + endorsement_boost)

def score_rules_context(c: dict) -> float:
    """Measure locations, background, and relocation preferences."""
    profile = c.get("profile", {})
    career_history = c.get("career_history", [])
    signals = c.get("redrob_signals", {})
    loc = profile.get("location", "").lower()
    
    score = 0.0
    
    # Product company history
    has_product_role = False
    for job in career_history:
        comp = job.get("company", "").lower()
        if comp and not any(cons in comp for cons in CONSULTING_COMPANIES):
            has_product_role = True
            break
    if has_product_role:
        score += 0.5
        
    # Location preference
    if any(city in loc for city in PREFERRED_CITIES):
        score += 0.3
        
    # Willing to relocate
    if signals.get("willing_to_relocate", False):
        score += 0.2
        
    # GitHub presence
    github_score = signals.get("github_activity_score", -1)
    if github_score > 0:
        score += min(0.2, github_score / 500.0)
        
    # HR-tech/marketplace background
    hrtech_keywords = {"hr tech", "recruiting", "talent", "marketplace", "staffing", "hr-tech"}
    for job in career_history:
        desc = job.get("description", "").lower()
        title = job.get("title", "").lower()
        if any(hk in desc or hk in title for hk in hrtech_keywords):
            score += 0.1
            break
            
    return min(1.0, score / 1.3)

def score_yoe(yoe: float) -> float:
    """Score YOE based on non-linear JD matching bands."""
    if 6.0 <= yoe <= 8.0:
        return 1.0
    elif 4.0 <= yoe < 6.0:
        return 0.75
    elif 8.0 < yoe <= 10.0:
        return 0.65
    elif yoe > 10.0:
        return 0.40
    else:
        return 0.20

# ============================================================================
# PENALTIES
# ============================================================================

def is_title_chaser(career_history: list) -> bool:
    """Check for frequent job changes (tenure under 18 months)."""
    if len(career_history) < 2:
        return False
    short_tenures = 0
    # check last 3 jobs
    for job in career_history[:3]:
        months = job.get("duration_months")
        if months is not None and months < 18:
            short_tenures += 1
    return short_tenures >= 2

def is_langchain_only(c: dict) -> bool:
    """Flag candidates with only LLM wrappers but no core ML training."""
    skills = c.get("skills", [])
    signals = c.get("redrob_signals", {})
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
    except:
        pass
    return False

def is_research_only(career_history: list) -> bool:
    """Identify pure researchers without any deployment experience."""
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

def calculate_penalties(c: dict) -> float:
    """Calculate the total additive penalties for a candidate."""
    profile = c.get("profile", {})
    career_history = c.get("career_history", [])
    yoe = profile.get("years_of_experience", 0.0)
    title = profile.get("current_title", "").lower()
    signals = c.get("redrob_signals", {})
    notice_days = signals.get("notice_period_days", 180)
    github_score = signals.get("github_activity_score", -1)
    
    penalties = 0.0
    
    # 1. Title chaser
    if is_title_chaser(career_history):
        penalties += 0.12
        
    # 2. Langchain only
    if is_langchain_only(c):
        penalties += 0.10
        
    # 3. Research only
    if is_research_only(career_history):
        penalties += 0.15
        
    # 4. Junior Research Engineer
    if "research" in title and yoe < 5.0:
        penalties += 0.25
        
    # 5. Notice period penalty
    if 30 < notice_days <= 60:
        penalties += 0.04
    elif 60 < notice_days <= 90:
        penalties += 0.08
    elif notice_days > 90:
        penalties += 0.12
        
    # 6. Consulting career penalty
    has_consulting = any(any(cons in job.get("company", "").lower() for cons in CONSULTING_COMPANIES) for job in career_history)
    all_consulting = career_history and all(any(cons in job.get("company", "").lower() for cons in CONSULTING_COMPANIES) for job in career_history)
    if all_consulting:
        penalties += 0.25
    elif has_consulting:
        penalties += 0.12
        
    # 7. Closed-source only
    if yoe >= 5.0 and github_score == -1:
        penalties += 0.08
        
    # 8. CV/Speech/Robotics primary (no NLP/IR evidence)
    cv_keywords = {"vision", "image", "speech", "audio", "robot", "robotics", "object detection", "segmentation", "video"}
    has_nlp_ir = False
    nlp_ir_keywords = {"nlp", "text", "search", "retrieval", "matching", "ranking", "relevance", "llm", "language", "semantic", "rag"}
    
    primary_title_is_cv = any(ck in title for ck in cv_keywords)
    
    for job in career_history:
        j_title = job.get("title", "").lower()
        j_desc = job.get("description", "").lower()
        if any(nk in j_title or nk in j_desc for nk in nlp_ir_keywords):
            has_nlp_ir = True
            
    if primary_title_is_cv and not has_nlp_ir:
        penalties += 0.10
        
    # 9. Over-experience penalty (12+ years of experience is too senior for the JD)
    if yoe > 12.0:
        penalties += 0.18
        
    return penalties

def is_consulting(company_name: str, industry: str) -> bool:
    if not company_name:
        return False
    company_lower = company_name.lower().strip()
    if any(c in company_lower for c in CONSULTING_COMPANIES):
        return True
    if industry and industry.lower() in ["it services", "it services and it consulting", "consulting"]:
        return True
    return False

# Backward compatibility alias
calculate_rules_score = score_rules_context

