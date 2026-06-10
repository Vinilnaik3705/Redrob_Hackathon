import streamlit as st
import pandas as pd
import subprocess
import tempfile
import os
import json

# Set up page styling
st.set_page_config(
    page_title="Redrob AI Candidate Ranker",
    page_icon="🎖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 10px;
    }
    .stButton>button {
        background-color: #3B82F6;
        color: white;
        font-weight: bold;
        border-radius: 6px;
        padding: 0.5rem 2rem;
        border: none;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #2563EB;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>Candidate Ranking System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>A hybrid semantic & rule-based pipeline for screening Senior AI Engineers</div>", unsafe_allow_html=True)

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
st.sidebar.write("Upload your candidate dataset in the main panel to begin.")

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
        # Show file info
        st.success(f"File uploaded successfully: {uploaded_file.name}")

st.write("---")

if uploaded_file is not None:
    if st.button("Run Candidate Ranking Engine"):
        with st.status("Processing candidates...", expanded=True) as status_box:
            # Create a temporary directory to store files
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_input_path = os.path.join(temp_dir, "candidates_input.jsonl")
                temp_output_path = os.path.join(temp_dir, "submission.csv")
                
                # Save uploaded file contents
                with open(temp_input_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                status_box.write("1. Saved upload. Running hard filters and computing embeddings...")
                
                # Run rank.py as a subprocess
                # Pass the modified JD if the user edited it
                # Note: We temporarily overwrite JD_QUERY in rank.py if needed,
                # but since it's defined inside rank.py, we can run it directly.
                # To support custom JDs, we could write it to a temp file, 
                # but running the default rank.py matches the official spec.
                
                cmd = [
                    "python", "rank.py",
                    "--candidates", temp_input_path,
                    "--out", temp_output_path,
                    "--jd", jd_query
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    status_box.write("2. Ranking completed successfully.")
                    status_box.update(label="Ranking complete!", state="complete")
                    
                    # Read results
                    if os.path.exists(temp_output_path):
                        df = pd.read_csv(temp_output_path)
                        
                        st.subheader("Top Ranked Candidates")
                        
                        # Show stats
                        stat_col1, stat_col2, stat_col3 = st.columns(3)
                        with stat_col1:
                            st.metric("Total Candidates Processed", len(df))
                        with stat_col2:
                            st.metric("Max Score", f"{df['score'].max():.4f}")
                        with stat_col3:
                            st.metric("Min Score", f"{df['score'].min():.4f}")
                        
                        # Download button
                        csv_data = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download team_The_Gladiators.csv",
                            data=csv_data,
                            file_name="team_The_Gladiators.csv",
                            mime="text/csv"
                        )
                        
                        # Interactive table
                        st.dataframe(
                            df,
                            column_config={
                                "rank": st.column_config.NumberColumn("Rank", format="%d"),
                                "candidate_id": "Candidate ID",
                                "score": st.column_config.NumberColumn("Score", format="%.4f"),
                                "reasoning": st.column_config.TextColumn("Reasoning Details", width="large")
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                else:
                    status_box.update(label="Ranking failed!", state="error")
                    st.error("Error executing ranking pipeline:")
                    st.code(result.stderr)
else:
    st.info("Please upload a candidates dataset to start the ranking process.")
