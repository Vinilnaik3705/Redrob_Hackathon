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

# Tiered skill taxonomy — weight core retrieval/embedding skills higher
JD_SKILLS_TIER1 = {
    "faiss", "pinecone", "qdrant", "weaviate", "milvus",
    "sentence-transformers", "bge", "e5", "embeddings", "vector databases"
}  # weight 3×

JD_SKILLS_TIER2 = {
    "rag", "lora", "qlora", "peft", "ndcg", "mrr", "map",
    "learning-to-rank", "elasticsearch", "opensearch",
    "hybrid search", "fine-tuning", "xgboost"
}  # weight 2×

JD_SKILLS_TIER3 = {
    "python", "pytorch", "tensorflow", "transformers",
    "numpy", "pandas", "scikit-learn", "docker", "kubernetes",
    "aws", "gcp", "azure", "git"
}  # weight 1×

# Negative signal skills — penalize clearly unrelated backgrounds
IRRELEVANT_SKILLS = {
    "photoshop", "illustrator", "figma", "canva",
    "marketing", "seo", "google ads", "facebook ads",
    "wordpress", "shopify", "wix", "squarespace",
    "accounting", "tally", "salesforce", "hubspot",
    "autocad", "solidworks", "catia", "revit"
}  # weight −0.5×


# Disqualified patterns for current title to eliminate unrelated applications
DISQUALIFIED_TITLE_PATTERNS = [
    r'\bdevops\b', r'\bsre\b', r'\bsite reliability\b',
    r'\bfrontend\b', r'\bfront-end\b', r'\bfront end\b',
    r'\bbackend developer\b', r'\bback-end developer\b',
    r'\bfull.?stack\b',
    r'\bproject manager\b', r'\bprogram manager\b',
    r'\bhr manager\b', r'\bhuman resources\b',
    r'\bmarketing manager\b', r'\bsales\b',
    r'\bmechanical engineer\b', r'\bcivil engineer\b',
    r'\bbusiness analyst\b', r'\bdata analyst\b',  # NOT data scientist
    r'\bcloud engineer\b', r'\bnetwork engineer\b',
    r'\baccountant\b', r'\bfinance\b',
]

def clean_text(text: str) -> str:
    """Clean and normalize whitespaces in text."""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text).strip()

def is_unrelated_role(candidate: dict) -> bool:
    title = candidate.get('profile', {}).get('current_title', '').lower()
    import re
    for pattern in DISQUALIFIED_TITLE_PATTERNS:
        if re.search(pattern, title):
            return True
            
    # Also check if title has ZERO AI/ML signal AND no AI in career
    AI_TITLE_SIGNALS = ['ml','ai','machine learning','data scientist','nlp',
                        'search','ranking','recommendation','applied scientist',
                        'research scientist','research engineer','scientist']
    has_ai_title = any(sig in title for sig in AI_TITLE_SIGNALS)
    
    if not has_ai_title:
        # Check career — if career also has no AI, disqualify
        career = candidate.get('career_history', [])
        career_titles = ' '.join(j.get('title','').lower() for j in career)
        career_descs = ' '.join(j.get('description','').lower() for j in career)
        combined = career_titles + ' ' + career_descs
        has_ai_career = any(sig in combined for sig in 
                           ['machine learning','deep learning','nlp','neural','embedding',
                            'vector','retrieval','ranking','recommendation','pytorch',
                            'tensorflow','transformer','llm','bert','gpt'])
        if not has_ai_career:
            return True
            
    return False

def get_active_days(signals: dict):
    # Try both field names
    days = signals.get('last_active_days_ago')
    if days is not None:
        return int(days)
        
    date_str = signals.get('last_active_date', '')
    if date_str:
        try:
            today = datetime(2026, 6, 7)
            return (today - datetime.strptime(date_str, "%Y-%m-%d")).days
        except:
            pass
    return None  # Unknown

def is_closed_source_only(candidate: dict) -> bool:
    profile = candidate.get('profile', {})
    yoe = profile.get('years_of_experience', 0.0)
    signals = candidate.get('redrob_signals', {})
    github = signals.get('github_activity_score', None)
    
    # Only flag if EXPLICITLY zero AND high YOE
    # None = field missing = unknown = do NOT penalize
    if yoe >= 7.0 and github is not None and float(github) == 0.0:
        return True
    return False

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
    country = profile.get("country", "").lower().strip()
    loc = profile.get("location", "").lower().strip()
    yoe = profile.get("years_of_experience", 0.0)
    signals = c.get("redrob_signals", {})
    
    # Bug 4 - Unrelated Title
    if is_unrelated_role(c):
        return True
        
    # Rule 9 - Outside India
    if country != "india" and "india" not in loc:
        return True
        
    # Rule 10 - Inactive > 12 Months
    active_days = get_active_days(signals)
    if active_days is not None and active_days > 365:
        return True
            
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
    return min(1.0, count / 8.0)

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
    return min(1.0, count / 8.0)

def score_behavioral_availability(c: dict) -> float:
    """Evaluate candidate availability using all 23 Redrob behavioral signals.
    Returns 0.0-1.0 where 1.0 = ideal candidate (active, responsive, available)."""
    signals = c.get("redrob_signals", {})
    
    # --- Recency & Activity (weight: 25%) ---
    days = get_active_days(signals)
    recency_score = 0.0
    if days is not None:
        if days <= 7:
            recency_score = 1.0
        elif days <= 180:
            recency_score = max(0.0, 1.0 - (days - 7) / 173.0)
    
    # --- Active job-seeking signals (weight: 20%) ---
    open_to_work = 1.0 if signals.get("open_to_work_flag", False) else 0.0
    apps_30d = min(1.0, signals.get("applications_submitted_30d", 0) / 10.0)
    seeking_score = open_to_work * 0.6 + apps_30d * 0.4
    
    # --- Responsiveness (weight: 20%) ---
    response_rate = float(signals.get("recruiter_response_rate", 0.0))
    avg_response_hrs = signals.get("avg_response_time_hours", 168)  # default 1 week
    response_time_score = 0.0
    if avg_response_hrs <= 12:
        response_time_score = 1.0
    elif avg_response_hrs <= 72:
        response_time_score = max(0.0, 1.0 - (avg_response_hrs - 12) / 60.0)
    responsiveness_score = response_rate * 0.6 + response_time_score * 0.4
    
    # --- Notice period (weight: 15%) ---
    notice_days = signals.get("notice_period_days", 180)
    if notice_days <= 0:
        notice_score = 1.0
    elif notice_days <= 15:
        notice_score = 0.95
    elif notice_days <= 30:
        notice_score = 0.80
    elif notice_days <= 60:
        notice_score = 0.55
    elif notice_days <= 90:
        notice_score = 0.30
    else:
        notice_score = 0.10
    
    # --- Market validation (weight: 10%) ---
    profile_views = min(1.0, signals.get("profile_views_received_30d", 0) / 20.0)
    saved_by = min(1.0, signals.get("saved_by_recruiters_30d", 0) / 5.0)
    search_appear = min(1.0, signals.get("search_appearance_30d", 0) / 30.0)
    market_score = (profile_views + saved_by + search_appear) / 3.0
    
    # --- Interview reliability (weight: 5%) ---
    interview_rate = float(signals.get("interview_completion_rate", 0.0))
    offer_rate = signals.get("offer_acceptance_rate", -1)
    if offer_rate == -1:
        offer_score = 0.5  # unknown — neutral
    else:
        offer_score = float(offer_rate)
    reliability_score = interview_rate * 0.6 + offer_score * 0.4
    
    # --- Profile trust (weight: 5%) ---
    completeness = signals.get("profile_completeness_score", 50) / 100.0
    verified = sum([
        1 if signals.get("verified_email", False) else 0,
        1 if signals.get("verified_phone", False) else 0,
        1 if signals.get("linkedin_connected", False) else 0,
    ]) / 3.0
    trust_score = completeness * 0.5 + verified * 0.5
    
    # --- Weighted combination ---
    raw = (
        0.25 * recency_score +
        0.20 * seeking_score +
        0.20 * responsiveness_score +
        0.15 * notice_score +
        0.10 * market_score +
        0.05 * reliability_score +
        0.05 * trust_score
    )
    return min(1.0, raw)

def score_skill_match(c: dict) -> float:
    """Trust-weighted, tiered skill scoring.
    Each skill is weighted by: tier_weight × trust_score.
    Trust = f(endorsements, duration_months, proficiency)."""
    skills = c.get("skills", [])
    if not skills:
        return 0.0
    
    # Build skill lookup with trust scores
    skill_map = {}  # name -> {trust, tier_weight}
    max_endorse = max((s.get("endorsements", 0) for s in skills), default=1) or 1
    max_duration = max((s.get("duration_months", 0) for s in skills), default=1) or 1
    
    for s in skills:
        name = s.get("name", "").lower().strip()
        endorse = s.get("endorsements", 0)
        duration = s.get("duration_months", 0)
        prof = s.get("proficiency", "beginner").lower()
        
        # Trust score: how much we believe this skill claim
        endorse_norm = min(1.0, endorse / max(max_endorse, 1))
        duration_norm = min(1.0, duration / max(max_duration, 1))
        prof_mult = {"expert": 1.0, "advanced": 0.85, "intermediate": 0.6, "beginner": 0.3}.get(prof, 0.5)
        
        trust = (endorse_norm * 0.4 + duration_norm * 0.4 + prof_mult * 0.2)
        
        # Determine tier weight
        if name in JD_SKILLS_TIER1:
            tier_weight = 3.0
        elif name in JD_SKILLS_TIER2:
            tier_weight = 2.0
        elif name in JD_SKILLS_TIER3:
            tier_weight = 1.0
        elif name in IRRELEVANT_SKILLS:
            tier_weight = -0.5
        else:
            tier_weight = 0.0  # not JD-related, neutral
        
        if tier_weight != 0.0:
            skill_map[name] = {"trust": trust, "tier_weight": tier_weight}
    
    # Calculate trust-weighted score
    positive_sum = sum(
        s["tier_weight"] * s["trust"]
        for s in skill_map.values() if s["tier_weight"] > 0
    )
    negative_sum = sum(
        abs(s["tier_weight"]) * (1.0 - s["trust"])  # less trusted irrelevant = worse
        for s in skill_map.values() if s["tier_weight"] < 0
    )
    
    # Normalize: max realistic ≈ 6 tier1×3×1.0 + 5 tier2×2×1.0 = 28
    tiered_score = min(1.0, max(0.0, (positive_sum - negative_sum) / 16.0))
    
    # Career corroboration — claimed skills appear in career descriptions
    skill_names = {s.get("name", "").lower().strip() for s in skills}
    all_matched = JD_REQUIRED_SKILLS.intersection(skill_names)
    career_history = c.get("career_history", [])
    career_text = " ".join([j.get("description", "").lower() + " " + j.get("title", "").lower() for j in career_history])
    
    corroborated = sum(1 for s in all_matched if s in career_text)
    corroboration_ratio = corroborated / max(len(all_matched), 1)
    
    # Combine: trust-weighted tiered score modulated by corroboration
    return min(1.0, tiered_score * (0.5 + 0.5 * corroboration_ratio))

def score_career_trajectory(c: dict) -> float:
    """Measure whether a candidate's career is growing into AI/retrieval or drifting away.
    Compares AI-relevance of recent roles vs early roles. Range: 0.0 to 1.0."""
    career_history = c.get("career_history", [])
    if len(career_history) < 2:
        return 0.5  # insufficient data — neutral
    
    # AI/retrieval relevance keywords
    ai_keywords = {
        "machine learning", "deep learning", "nlp", "natural language",
        "search", "ranking", "retrieval", "recommendation", "embedding",
        "vector", "neural", "transformer", "llm", "bert", "gpt",
        "pytorch", "tensorflow", "ml engineer", "ai engineer",
        "data scientist", "applied scientist", "research scientist",
        "information retrieval", "reranking", "faiss", "pinecone",
        "qdrant", "weaviate", "milvus", "rag", "fine-tuning",
        "production", "deployed", "shipped", "serving", "inference"
    }
    
    def job_relevance(job):
        """Score a single job's AI/retrieval relevance (0-1)."""
        title = job.get("title", "").lower()
        desc = job.get("description", "").lower()
        combined = title + " " + desc
        hits = sum(1 for kw in ai_keywords if kw in combined)
        return min(1.0, hits / 5.0)
    
    # Score each job
    scores = [job_relevance(job) for job in career_history]
    
    # Split into first half (early career) and second half (recent career)
    mid = len(scores) // 2
    if mid == 0:
        mid = 1
    early_avg = sum(scores[mid:]) / len(scores[mid:])   # later in list = earlier career (usually)
    recent_avg = sum(scores[:mid]) / len(scores[:mid])   # first in list = current/recent
    
    # Trajectory: positive = growing into AI, negative = drifting away
    trajectory = recent_avg - early_avg  # range: -1.0 to +1.0
    
    # Also penalize title-chaser pattern (avg tenure < 15 months in last 3 roles)
    recent_jobs = career_history[:3]
    tenures = [j.get("duration_months", 24) for j in recent_jobs if j.get("duration_months") is not None]
    if tenures:
        avg_tenure = sum(tenures) / len(tenures)
        if avg_tenure < 12:
            trajectory -= 0.25  # severe hopper
        elif avg_tenure < 18:
            trajectory -= 0.10  # mild hopper
    
    # Map trajectory from [-1.25, +1.0] to [0.0, 1.0]
    # -1.25 -> 0.0, 0.0 -> 0.55, +1.0 -> 1.0
    normalized = max(0.0, min(1.0, (trajectory + 1.25) / 2.25))
    
    # Boost if current role itself is highly relevant
    if scores and scores[0] >= 0.6:
        normalized = min(1.0, normalized + 0.1)
    
    return normalized

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
    """Identify researchers without any deployment experience."""
    if not career_history:
        return False
    research_titles = ["research scientist", "research engineer", "research intern", "phd student", "postdoc", "research fellow", "ml researcher", "ai researcher", "scientist"]
    has_research_title = any(any(rt in job.get("title", "").lower() for rt in research_titles) for job in career_history)
    
    production_keywords = ["production", "deployed", "shipped", "launched", "real users", "serving", "inference"]
    has_production_signal = False
    for job in career_history:
        title = job.get("title", "").lower()
        desc = job.get("description", "").lower()
        if any(pk in desc or pk in title for pk in production_keywords):
            has_production_signal = True
            
    return has_research_title and not has_production_signal

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
    if is_closed_source_only(c):
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

def score_assessment(c: dict) -> float:
    """Score based on platform skill assessment results from redrob_signals.
    Verified assessments are stronger signals than self-reported skills."""
    signals = c.get("redrob_signals", {})
    assessments = signals.get("skill_assessments", [])
    if not assessments:
        return 0.0
    
    # Weight assessments by relevance to the JD
    jd_relevant_terms = {
        "sentence transformers", "embeddings", "vector", "faiss", "pinecone",
        "retrieval", "search", "ranking", "rag", "nlp", "nlu",
        "machine learning", "deep learning", "pytorch", "tensorflow",
        "python", "transformers", "bert", "gpt", "llm"
    }
    
    total_score = 0.0
    relevant_count = 0
    general_count = 0
    
    for a in assessments:
        ascore = float(a.get("score", 0))
        skill_name = a.get("skill", "").lower()
        
        # Check if assessment is for a JD-relevant skill
        is_relevant = any(term in skill_name for term in jd_relevant_terms)
        
        if is_relevant:
            total_score += ascore * 1.5  # 1.5× weight for relevant assessments
            relevant_count += 1
        else:
            total_score += ascore
            general_count += 1
    
    count = relevant_count + general_count
    if count == 0:
        return 0.0
    
    avg_score = total_score / (relevant_count * 1.5 + general_count) if (relevant_count * 1.5 + general_count) > 0 else 0.0
    
    # Normalize to 0-1 range (assessment scores are typically 0-100)
    normalized = min(1.0, avg_score / 100.0)
    
    # Bonus for having multiple relevant assessments
    multi_bonus = min(0.15, relevant_count * 0.05)
    
    return min(1.0, normalized + multi_bonus)


# Backward compatibility alias
calculate_rules_score = score_rules_context
