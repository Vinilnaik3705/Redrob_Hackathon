import streamlit as st
import pandas as pd
import subprocess
import tempfile
import os
import json

# Set up page styling
st.set_page_config(
    page_title="Redrob AI Candidate Ranker & Explorer",
    page_icon="🎖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium styling
st.markdown("""
    <style>
    /* Theme override & premium typography */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 2.8rem;
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        font-size: 1.15rem;
        color: #4B5563;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Premium Cards */
    .premium-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        border: 1px solid #E5E7EB;
        margin-bottom: 15px;
        transition: transform 0.2s, box-shadow 0.2s;
        color: #1F2937 !important;
    }
    .premium-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.08);
    }
    .premium-card p, .premium-card h3, .premium-card h4, .premium-card li, .premium-card b, .premium-card span {
        color: #1F2937 !important;
    }
    
    /* Badges */
    .badge {
        padding: 4px 10px;
        border-radius: 50px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 5px;
        margin-bottom: 5px;
        display: inline-block;
    }
    .badge-primary { background-color: #E0E7FF; color: #4338CA; border: 1px solid #C7D2FE; }
    .badge-success { background-color: #D1FAE5; color: #065F46; border: 1px solid #A7F3D0; }
    .badge-warning { background-color: #FEF3C7; color: #92400E; border: 1px solid #FDE68A; }
    .badge-danger { background-color: #FEE2E2; color: #991B1B; border: 1px solid #FCA5A5; }
    .badge-info { background-color: #E0F2FE; color: #0369A1; border: 1px solid #BAE6FD; }
    .badge-neutral { background-color: #F3F4F6; color: #374151; border: 1px solid #E5E7EB; }
    
    /* Skill Proficiencies */
    .skill-expert { background-color: #3B82F6; color: white; }
    .skill-advanced { background-color: #10B981; color: white; }
    .skill-intermediate { background-color: #F59E0B; color: white; }
    .skill-beginner { background-color: #9CA3AF; color: white; }
    
    /* Metrics box */
    .stat-container {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border: 1px solid #BFDBFE;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
    }
    .stat-container h3 {
        color: #1E3A8A !important;
        margin: 0 0 5px 0 !important;
        font-size: 2.2rem !important;
        font-weight: 700 !important;
    }
    .stat-container p {
        color: #4B5563 !important;
        margin: 0 !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
    }
    
    /* Button layout */
    .stButton>button {
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem 2.5rem;
        border: none;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.2);
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2563EB 0%, #1E40AF 100%);
        box-shadow: 0 6px 12px rgba(59, 130, 246, 0.3);
        transform: translateY(-1px);
    }
    
    /* Career history entry */
    .timeline-item {
        border-left: 3px solid #3B82F6;
        padding-left: 20px;
        margin-left: 10px;
        position: relative;
        margin-bottom: 20px;
    }
    .timeline-item::before {
        content: '';
        width: 12px;
        height: 12px;
        background-color: #3B82F6;
        border-radius: 50%;
        position: absolute;
        left: -8px;
        top: 5px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>Redrob AI Candidate Ranker</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>A premium hybrid semantic & rule-based pipeline for screening Senior AI Engineers</div>", unsafe_allow_html=True)

# Default Job Description Query
DEFAULT_JD = (
    "Senior AI Engineer with production experience in embeddings-based retrieval systems "
    "(sentence-transformers, BGE, E5, OpenAI embeddings), vector databases and hybrid search "
    "(Pinecone, Weaviate, Qdrant, Milvus, FAISS, Elasticsearch, OpenSearch), Python, "
    "ranking evaluation frameworks (NDCG, MRR, MAP, A/B testing, offline-to-online correlation). "
    "LLM fine-tuning (LoRA, QLoRA, PEFT), learning-to-rank. "
    "Startup product company experience. Located in India (Noida, Pune, Hyderabad, Bangalore, Mumbai, Delhi NCR)."
)

# Sidebar configuration
st.sidebar.header("Configuration")
st.sidebar.write("Configure the ranking system options below.")
candidate_limit = st.sidebar.slider(
    "Number of Candidates to Output",
    min_value=10,
    max_value=500,
    value=100,
    step=10,
    help="Select the maximum number of ranked candidates to return in the CSV output."
)

# Initialize Session State
if 'results_df' not in st.session_state:
    st.session_state['results_df'] = None
if 'candidates_details' not in st.session_state:
    st.session_state['candidates_details'] = {}
if 'temp_input_path' not in st.session_state:
    st.session_state['temp_input_path'] = None

# Main layout split
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("Job Description (JD)")
    jd_query = st.text_area(
        "Edit the query text:",
        value=DEFAULT_JD,
        height=180
    )

with col2:
    st.subheader("Upload Candidates Dataset")
    uploaded_file = st.file_uploader(
        "Upload a JSON or JSONL file of candidates:",
        type=["json", "jsonl"],
        help="Upload the raw candidate profiles dataset."
    )
    
    if uploaded_file is not None:
        st.success(f"File uploaded successfully: {uploaded_file.name}")

st.write("---")

if uploaded_file is not None:
    if st.button("Run Candidate Ranking Engine"):
        with st.status("Processing candidates...", expanded=True) as status_box:
            # Create a temporary directory to store files
            temp_dir = tempfile.mkdtemp()
            temp_input_path = os.path.join(temp_dir, "candidates_input.jsonl")
            temp_output_path = os.path.join(temp_dir, "submission.csv")
            
            # Save uploaded file contents
            with open(temp_input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.session_state['temp_input_path'] = temp_input_path
            
            status_box.write("1. Saved upload. Parsing candidate metadata...")
            
            # Parse candidate details
            candidates_details = {}
            try:
                with open(temp_input_path, "r", encoding="utf-8") as f:
                    first_char = f.read(1)
                    f.seek(0)
                    if first_char == '[':
                        data = json.load(f)
                        for item in data:
                            if "candidate_id" in item:
                                candidates_details[item["candidate_id"]] = item
                    else:
                        for line in f:
                            if line.strip():
                                try:
                                    item = json.loads(line)
                                    if "candidate_id" in item:
                                        candidates_details[item["candidate_id"]] = item
                                except Exception:
                                    pass
                st.session_state['candidates_details'] = candidates_details
            except Exception as e:
                st.error(f"Error parsing candidates file: {e}")
            
            status_box.write("2. Running ranking engine subprocess (embeddings, rules, and cross-encoder)...")
            
            import sys
            app_dir = os.path.dirname(os.path.abspath(__file__))
            rank_script = os.path.join(app_dir, "rank.py")
            cmd = [
                sys.executable, rank_script,
                "--candidates", temp_input_path,
                "--out", temp_output_path,
                "--jd", jd_query,
                "--limit", str(candidate_limit)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=app_dir)
            
            if result.returncode == 0:
                status_box.write("3. Ranking completed successfully.")
                status_box.update(label="Ranking complete!", state="complete")
                
                if os.path.exists(temp_output_path):
                    df = pd.read_csv(temp_output_path)
                    st.session_state['results_df'] = df
            else:
                status_box.update(label="Ranking failed!", state="error")
                st.error("Error executing ranking pipeline:")
                if result.stderr:
                    st.markdown("### Stderr Output:")
                    st.code(result.stderr)
                if result.stdout:
                    st.markdown("### Stdout Output:")
                    st.code(result.stdout)

# Display Results & Interactive Features if available
if st.session_state['results_df'] is not None:
    df = st.session_state['results_df'].copy()
    details = st.session_state['candidates_details']
    
    # Enrich df with details from details dictionary if available
    for col in ['name', 'title', 'yoe', 'location', 'notice_period', 'willing_to_relocate', 'skills_count']:
        if col not in df.columns:
            df[col] = None
            
    for idx, row in df.iterrows():
        cid = row['candidate_id']
        if cid in details:
            cand = details[cid]
            p = cand.get('profile', {})
            sig = cand.get('redrob_signals', {})
            df.at[idx, 'name'] = p.get('anonymized_name', cid)
            df.at[idx, 'title'] = p.get('current_title', 'Unknown Title')
            df.at[idx, 'yoe'] = p.get('years_of_experience', 0.0)
            df.at[idx, 'location'] = p.get('location', 'India')
            df.at[idx, 'notice_period'] = sig.get('notice_period_days', 180)
            df.at[idx, 'willing_to_relocate'] = "Yes" if sig.get('willing_to_relocate', False) else "No"
            df.at[idx, 'skills_count'] = len(cand.get('skills', []))
            
    # Identify disqualified candidates
    ranked_ids = set(df['candidate_id'].tolist())
    disqualified_candidates = []
    
    from scorer import check_honeypots, is_unrelated_role, is_research_only, get_active_days
    
    for cid, cand in details.items():
        if cid not in ranked_ids:
            p = cand.get('profile', {})
            sig = cand.get('redrob_signals', {})
            
            # Diagnose reason
            if check_honeypots(cand):
                reason = "Disqualified: Honeypot Trap (Fake Profile/Timelines)"
            elif is_unrelated_role(cand):
                reason = f"Disqualified: Unrelated Role ('{p.get('current_title', 'Unknown')}')"
            elif p.get('country', '').lower().strip() != 'india' and 'india' not in p.get('location', '').lower():
                reason = f"Disqualified: Location Outside India ('{p.get('location', 'Unknown')}')"
            elif get_active_days(sig) is not None and get_active_days(sig) > 365:
                reason = f"Disqualified: Inactive > 12 Months ({get_active_days(sig)} days)"
            elif p.get('years_of_experience', 0.0) < 3.0:
                reason = f"Disqualified: YOE < 3.0 ({p.get('years_of_experience', 0.0)} yrs)"
            elif is_research_only(cand.get('career_history', [])):
                reason = "Disqualified: Pure Research (No production deployment)"
            else:
                reason = "Filtered out: Low semantic/technical match score (0.0)"
                
            disqualified_candidates.append({
                "candidate_id": cid,
                "name": p.get('anonymized_name', cid),
                "title": p.get('current_title', 'Unknown Title'),
                "yoe": p.get('years_of_experience', 0.0),
                "location": p.get('location', 'India'),
                "reason": reason
            })
            
    # Setup interactive tabs
    tab_leaderboard, tab_explorer, tab_analytics, tab_compare = st.tabs([
        "🏆 Ranked Leaderboard", 
        "🔍 Profile Explorer", 
        "📊 Talent Analytics", 
        "⚖️ Compare Candidates"
    ])
    
    # --- TAB 1: RANKED LEADERBOARD ---
    with tab_leaderboard:
        st.subheader("Leaderboard Results")
        
        # Sub-tabs for Shortlisted vs Disqualified
        sub_tab_shortlist, sub_tab_disqualified = st.tabs(["🏆 Shortlisted Candidates", "🚫 Disqualified / Filtered Candidates"])
        
        with sub_tab_shortlist:
            # Summary metrics
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            with stat_col1:
                st.markdown(f"<div class='stat-container'><h3>{len(df)}</h3><p style='margin:0;color:#6B7280;'>Total Shortlisted</p></div>", unsafe_allow_html=True)
            with stat_col2:
                max_score = df['score'].max() if len(df) > 0 else 0.0
                st.markdown(f"<div class='stat-container'><h3>{max_score:.4f}</h3><p style='margin:0;color:#6B7280;'>Highest Score</p></div>", unsafe_allow_html=True)
            with stat_col3:
                min_score = df['score'].min() if len(df) > 0 else 0.0
                st.markdown(f"<div class='stat-container'><h3>{min_score:.4f}</h3><p style='margin:0;color:#6B7280;'>Lowest Score</p></div>", unsafe_allow_html=True)
            with stat_col4:
                avg_yoe = df['yoe'].mean() if len(df) > 0 else 0.0
                st.markdown(f"<div class='stat-container'><h3>{avg_yoe:.1f} yrs</h3><p style='margin:0;color:#6B7280;'>Average YOE</p></div>", unsafe_allow_html=True)
                
            st.write("")
            
            # Download and Sidebar filters for Table view
            dl_col, filter_col = st.columns([1, 3])
            with dl_col:
                csv_data = st.session_state['results_df'].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download submission.csv",
                    data=csv_data,
                    file_name="team_The_Gladiators.csv",
                    mime="text/csv"
                )
                
            # Table filters
            st.markdown("### Filter Candidates List")
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                min_yoe_filter = st.slider("Minimum YOE", min_value=3.0, max_value=15.0, value=3.0, step=0.5)
            with f_col2:
                max_notice_filter = st.selectbox("Max Notice Period (Days)", options=[180, 90, 60, 30, 15, 0], index=0)
            with f_col3:
                relocate_only = st.checkbox("Willing to Relocate Only", value=False)
                
            # Apply filters
            if len(df) > 0:
                filtered_df = df[df['yoe'] >= min_yoe_filter]
                filtered_df = filtered_df[filtered_df['notice_period'] <= max_notice_filter]
                if relocate_only:
                    filtered_df = filtered_df[filtered_df['willing_to_relocate'] == "Yes"]
            else:
                filtered_df = df
                
            st.dataframe(
                filtered_df[['rank', 'candidate_id', 'name', 'title', 'yoe', 'location', 'notice_period', 'score', 'reasoning']],
                column_config={
                    "rank": st.column_config.NumberColumn("Rank", format="%d"),
                    "candidate_id": "ID",
                    "name": "Candidate Name",
                    "title": "Current Title",
                    "yoe": st.column_config.NumberColumn("YOE", format="%.1f"),
                    "location": "Location",
                    "notice_period": st.column_config.NumberColumn("Notice (Days)", format="%d"),
                    "score": st.column_config.NumberColumn("Match Score", format="%.4f"),
                    "reasoning": st.column_config.TextColumn("Summary / Fit Notes", width="large")
                },
                hide_index=True,
                use_container_width=True
            )
            
        with sub_tab_disqualified:
            st.markdown("### Disqualified & Filtered Candidates")
            st.markdown("These candidates did not make the leaderboard. Below is the Stage 1 Hard Filter / Honeypot diagnosis:")
            if disqualified_candidates:
                dis_df = pd.DataFrame(disqualified_candidates)
                st.dataframe(
                    dis_df,
                    column_config={
                        "candidate_id": "ID",
                        "name": "Candidate Name",
                        "title": "Current Title",
                        "yoe": st.column_config.NumberColumn("YOE", format="%.1f"),
                        "location": "Location",
                        "reason": st.column_config.TextColumn("Disqualification Reason", width="large")
                    },
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.success("No candidates were disqualified in this run!")
        
    # --- TAB 2: PROFILE EXPLORER ---
    with tab_explorer:
        st.subheader("Interactive Profile Inspector")
        selected_cid = st.selectbox("Select a candidate to inspect details:", options=df['candidate_id'].tolist())
        
        if selected_cid and selected_cid in details:
            cand = details[selected_cid]
            p = cand.get('profile', {})
            sig = cand.get('redrob_signals', {})
            career = cand.get('career_history', [])
            skills = cand.get('skills', [])
            edu = cand.get('education', [])
            
            # Find the row in the scoring dataframe
            score_row = df[df['candidate_id'] == selected_cid].iloc[0]
            
            st.write("---")
            
            # Main Details Grid
            det_col1, det_col2 = st.columns([1, 2])
            
            with det_col1:
                st.markdown(f"""
                    <div class='premium-card'>
                        <h3 style='margin:0 0 5px 0;'>{p.get('anonymized_name', selected_cid)}</h3>
                        <p style='color:#4B5563; font-weight:600; margin:0 0 15px 0;'>{p.get('current_title', 'Engineer')}</p>
                        <p>📍 <b>Location:</b> {p.get('location', 'India')}</p>
                        <p>💼 <b>Company:</b> {p.get('current_company', 'N/A')}</p>
                        <p>⏱️ <b>Notice Period:</b> {sig.get('notice_period_days', 'N/A')} days</p>
                        <p>💸 <b>Salary Range (INR):</b> {sig.get('expected_salary_range_inr_lpa', {}).get('min', 'N/A')}L - {sig.get('expected_salary_range_inr_lpa', {}).get('max', 'N/A')}L LPA</p>
                        <p>🤝 <b>Willing to relocate:</b> {"Yes" if sig.get('willing_to_relocate', False) else "No"}</p>
                        <p>💻 <b>Work Mode Preference:</b> {sig.get('preferred_work_mode', 'N/A').capitalize()}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Behavioral & Trust Metrics
                completeness = sig.get('profile_completeness_score', 0.0)
                github_score = sig.get('github_activity_score', -1)
                git_display = f"{github_score:.1f}" if github_score > 0 else "N/A"
                
                st.markdown(f"""
                    <div class='premium-card' style='background-color:#F9FAFB;'>
                        <h4 style='margin-top:0;'>Redrob System Signals</h4>
                        <p>🎯 <b>Rank Position:</b> #{score_row['rank']}</p>
                        <p>📈 <b>Final Match Score:</b> {score_row['score']:.4f}</p>
                        <p>⭐ <b>Profile Completeness:</b> {completeness}%</p>
                        <p>🐙 <b>GitHub Activity Score:</b> {git_display}</p>
                        <p>📧 <b>Verified Email:</b> {"✅" if sig.get('verified_email', False) else "❌"}</p>
                        <p>📱 <b>Verified Phone:</b> {"✅" if sig.get('verified_phone', False) else "❌"}</p>
                        <p>🔗 <b>LinkedIn Connected:</b> {"✅" if sig.get('linkedin_connected', False) else "❌"}</p>
                    </div>
                """, unsafe_allow_html=True)
                
            with det_col2:
                # Headline and Summary
                st.markdown(f"**Headline:** *{p.get('headline', 'N/A')}*")
                st.markdown(f"**Professional Summary:**\n> {p.get('summary', 'No summary provided')}")
                
                st.write("")
                st.subheader("Reasoning / Fit Assessment")
                st.info(score_row['reasoning'])
                
                # Skills Display
                st.subheader("Skills & Endorsements")
                if skills:
                    # Sort skills by proficiency level
                    prof_order = {"expert": 0, "advanced": 1, "intermediate": 2, "beginner": 3}
                    sorted_skills = sorted(skills, key=lambda s: prof_order.get(s.get('proficiency', 'beginner').lower(), 4))
                    
                    skills_html = ""
                    for s in sorted_skills:
                        name = s.get('name', '')
                        prof = s.get('proficiency', 'beginner').lower()
                        endorse = s.get('endorsements', 0)
                        dur = s.get('duration_months', 0)
                        
                        prof_class = f"skill-{prof}"
                        dur_text = f" ({dur}m)" if dur > 0 else ""
                        skills_html += f"<span class='badge {prof_class}'>{name} (Lvl: {prof.capitalize()}{dur_text})</span> "
                    
                    st.markdown(skills_html, unsafe_allow_html=True)
                else:
                    st.write("No skill tags listed.")
            
            st.write("---")
            
            # Education and Career History
            edu_col, career_col = st.columns([1, 2])
            
            with edu_col:
                st.subheader("🎓 Education")
                if edu:
                    for e in edu:
                        st.markdown(f"""
                            <div class='premium-card' style='padding:15px; margin-bottom:10px;'>
                                <b>{e.get('degree', 'Degree')} in {e.get('field_of_study', 'Field')}</b><br>
                                <span style='color:#4B5563;'>{e.get('institution', 'University')}</span><br>
                                <span class='badge badge-info'>Tier: {e.get('tier', 'Unknown').upper()}</span>
                                <span class='badge badge-neutral'>{e.get('start_year', '')} - {e.get('end_year', '')}</span>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.write("No education details listed.")
                    
            with career_col:
                st.subheader("💼 Career History")
                if career:
                    for job in career:
                        dur_y = f"{job.get('duration_months', 0) // 12}y"
                        dur_m = f"{job.get('duration_months', 0) % 12}m"
                        duration_str = f"{dur_y} {dur_m}" if job.get('duration_months') else "N/A"
                        
                        desc_text = job.get('description', '')
                        
                        st.markdown(f"""
                            <div class='timeline-item'>
                                <h4 style='margin:0;'>{job.get('title', 'Engineer')}</h4>
                                <p style='color:#4B5563; font-weight:600; margin:3px 0;'>{job.get('company', 'Company')} | <span style='font-size:0.85rem; font-weight:normal;'>{job.get('start_date', '')} to {job.get('end_date') or 'Present'} ({duration_str})</span></p>
                                <p style='font-size:0.9rem; color:#374151;'>{desc_text}</p>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.write("No career history listed.")
                    
    # --- TAB 3: TALENT ANALYTICS ---
    with tab_analytics:
        st.subheader("Top Talent Analytics & Demographics")
        
        graph_col1, graph_col2 = st.columns(2)
        
        with graph_col1:
            st.markdown("#### Score Distribution")
            score_data = df['score'].round(2).value_counts().sort_index().reset_index()
            score_data.columns = ['Score', 'Count']
            st.bar_chart(score_data, x='Score', y='Count')
            
            st.markdown("#### Experience Level Distribution (YOE)")
            yoe_data = df['yoe'].round(0).value_counts().sort_index().reset_index()
            yoe_data.columns = ['YOE', 'Count']
            st.bar_chart(yoe_data, x='YOE', y='Count')
            
        with graph_col2:
            st.markdown("#### Notice Period Breakdown (Days)")
            notice_data = df['notice_period'].value_counts().sort_index().reset_index()
            notice_data.columns = ['Notice Period (Days)', 'Count']
            st.bar_chart(notice_data, x='Notice Period (Days)', y='Count')
            
            st.markdown("#### Location Analysis")
            loc_data = df['location'].apply(lambda x: x.split(',')[0].strip() if isinstance(x, str) else 'India').value_counts().head(10).reset_index()
            loc_data.columns = ['Location', 'Count']
            st.bar_chart(loc_data, x='Location', y='Count')
            
    # --- TAB 4: COMPARE CANDIDATES ---
    with tab_compare:
        st.subheader("Side-by-Side Comparison")
        st.write("Select up to 3 candidates to compare their credentials side-by-side:")
        
        selected_compare = st.multiselect(
            "Select candidates to compare:",
            options=df['candidate_id'].tolist(),
            default=df['candidate_id'].head(2).tolist()[:3]
        )
        
        if len(selected_compare) > 0:
            compare_cols = st.columns(len(selected_compare))
            for i, cid in enumerate(selected_compare):
                with compare_cols[i]:
                    cand = details.get(cid, {})
                    p = cand.get('profile', {})
                    sig = cand.get('redrob_signals', {})
                    score_row = df[df['candidate_id'] == cid].iloc[0]
                    
                    st.markdown(f"""
                        <div class='premium-card' style='border-top: 5px solid #3B82F6;'>
                            <h3 style='margin:0;'>#{score_row['rank']}: {p.get('anonymized_name', cid)}</h3>
                            <h4 style='color:#10B981; margin: 5px 0 15px 0;'>Match Score: {score_row['score']:.4f}</h4>
                            <p>💼 <b>Title:</b> {p.get('current_title', 'N/A')}</p>
                            <p>🏢 <b>Company:</b> {p.get('current_company', 'N/A')}</p>
                            <p>🎓 <b>Experience:</b> {p.get('years_of_experience', 0.0)} yrs</p>
                            <p>⏱️ <b>Notice Period:</b> {sig.get('notice_period_days', 'N/A')} days</p>
                            <p>💸 <b>Salary expected:</b> {sig.get('expected_salary_range_inr_lpa', {}).get('min', 'N/A')}L - {sig.get('expected_salary_range_inr_lpa', {}).get('max', 'N/A')}L LPA</p>
                            <p>📍 <b>Location:</b> {p.get('location', 'N/A')}</p>
                            <p>🐙 <b>GitHub Score:</b> {sig.get('github_activity_score', 'N/A')}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("**Core Fit Summary:**")
                    st.info(score_row['reasoning'])
                    
                    st.markdown("**Recent Roles:**")
                    for job in cand.get('career_history', [])[:2]:
                        st.markdown(f"- **{job.get('title')}** at *{job.get('company')}* ({job.get('duration_months', 0)} months)")
                    
                    st.markdown("**Education:**")
                    for e in cand.get('education', [])[:1]:
                        st.markdown(f"- **{e.get('degree')}** from *{e.get('institution')}* (Tier {e.get('tier', 'N/A').upper()})")
                        
                    st.markdown("**Key Skills:**")
                    skills_list = [s.get('name') for s in cand.get('skills', [])]
                    st.write(", ".join(skills_list[:8]) + "...")
        else:
            st.info("Please select at least one candidate to compare.")
else:
    st.info("Please upload a candidates dataset and run the ranking process to visualize detailed results.")
