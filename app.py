# """Reddit Sentiment Analytics Pro Made By https://github.com/SachinMallah """

# from __future__ import annotations
# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import re
# import time
# import nltk
# import pandas as pd
# import streamlit as st
# import plotly.express as px
# import plotly.graph_objects as go
# from pathlib import Path
# from typing import Dict, List, Tuple, Optional
# from datetime import datetime
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from nltk.sentiment import SentimentIntensityAnalyzer
# from dataclasses import dataclass, field
# from wordcloud import WordCloud
# from loguru import logger
# from src.reddit_client import RedditClient


# # CONFIGURATION
# @dataclass(frozen=True)
# class AppConfig:
#     min_comment_length: int = 15
#     max_comment_length: int = 512
#     vader_workers: int = os.cpu_count() * 4
#     color_scheme: Dict[str, str] = field(
#         default_factory=lambda: {"Negative": "#EF553B", "Neutral": "#636EFA", "Positive": "#00CC96"}
#     )
#     safety_margin: float = 0.2
#     wordcloud_max_words: int = 200
#     sentiment_bins: Tuple[float] = (-1.0, -0.25, 0.25, 1.0)


# # PERFORMANCE OPTIMIZATIONS
# sys.path.append(str(Path(__file__).parent.parent))
# nltk.download('vader_lexicon', quiet=True)


# # ENHANCED STREAMLIT UI
# def configure_ui() -> None:
    
#     st.set_page_config(
#         page_title="Reddit Analytics Pro",
#         layout="wide",
#         page_icon="📊",
#         initial_sidebar_state="expanded",
#         menu_items={
#             'Get Help': 'https://github.com/SachinMallah/Reddit-Sentiment-Analysis',
#             'Report a bug': "mailto: Sachinmallah118@gmail.com",
#             'About': "### Reddit Sentiment Analytics"
#         }
#     )
    
#     st.markdown("""
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
#     .stApp { 
#         font-family: 'Inter', sans-serif; 
#         background: #0f0f0f; 
#         color: #ffffff;
#     }
#     .st-emotion-cache-1dp5vir { 
#         background: linear-gradient(90deg, #00cc96 0%, #636efa 100%);
#     }
#     .element:hover { 
#         transform: translateY(-3px); 
#         transition: all 0.3s ease;
#     }
#     @media (max-width: 768px) { 
#         .responsive-grid { 
#             grid-template-columns: 1fr !important; 
#         }
#     }
#     </style>
#     """, unsafe_allow_html=True)


# # ERROR HANDLING SYSTEM
# class AnalysisError(Exception):
    
#     pass

# class DataProcessingError(AnalysisError):
#     pass

# def handle_error(e: Exception, context: str = "") -> None:
    
#     error_id = f"ERR-{int(time.time())}"
#     logger.error(f"{error_id} - {context}: {str(e)}")
    
#     error_template = f"""
#     <div style="border-left: 4px solid #EF553B; padding: 1rem; background: #1a1a1a;">
#         <h3 style="color: #EF553B;">Error {error_id}</h3>
#         <p><strong>Context:</strong> {context}</p>
#         <p><strong>Details:</strong> {str(e)}</p>
#         <p><small>{datetime.now().isoformat()}</small></p>
#     </div>
#     """
#     st.markdown(error_template, unsafe_allow_html=True)
#     st.session_state.last_error = error_id
#     st.stop()


# # PARALLEL PROCESSING ENGINE
# def parallel_vader(texts: List[str], cfg: AppConfig) -> List[float]:
    
#     analyzer = SentimentIntensityAnalyzer()
#     validated_texts = [str(t)[:cfg.max_comment_length] if isinstance(t, (str, bytes)) else '' for t in texts]
    
#     with ThreadPoolExecutor(max_workers=cfg.vader_workers) as executor:
#         futures = {executor.submit(analyzer.polarity_scores, t): i for i, t in enumerate(validated_texts)}
        
#         results = [0.0] * len(texts)
#         for future in as_completed(futures):
#             idx = futures[future]
#             try:
#                 results[idx] = future.result()["compound"]
#             except Exception as e:
#                 logger.warning(f"VADER processing failed for index {idx}: {str(e)}")
#                 results[idx] = 0.0
                
#         return results

# def vectorized_clean(text_series: pd.Series, cfg: AppConfig) -> pd.Series:
    
#     return (
#         text_series.astype(str)
#         .str[:cfg.max_comment_length]
#         .str.lower()
#         .str.replace(r'[^\w\s]', '', regex=True)
#         .str.strip()
#         .fillna('')
#     )


# # CORE PROCESSING PIPELINE
# @st.cache_data(ttl=300, show_spinner=False)
# def analyze_comments(cfg: AppConfig, keyword: str, post_limit: int) -> pd.DataFrame:
    
#     try:
#         # Phase 1: Data Acquisition
#         with st.spinner("🌐 Fetching Reddit Data..."):
#             client = RedditClient()
#             df = client.fetch_comments(keyword, post_limit)
#             if df.empty:
#                 return pd.DataFrame()

#         # Phase 2: Robust Data Preparation
#         df["clean_text"] = vectorized_clean(df["body"], cfg)
#         valid_comments = df[df["clean_text"].str.len() >= cfg.min_comment_length]
        
#         if valid_comments.empty:
#             return pd.DataFrame()

#         # Phase 3: Parallel Sentiment Analysis
#         with st.spinner("🧠 Analyzing Sentiment..."):
#             scores = parallel_vader(valid_comments["clean_text"].tolist(), cfg)
#             valid_comments = valid_comments.assign(sentiment=scores)

#         # Phase 4: Post-processing
#         valid_comments["sentiment_label"] = pd.cut(
#             valid_comments["sentiment"],
#             bins=cfg.sentiment_bins,
#             labels=["Negative", "Neutral", "Positive"]
#         ).fillna("Neutral")
        
#         validate_results(valid_comments)
#         return valid_comments

#     except Exception as e:
#         handle_error(e, "Core processing pipeline")

# def validate_results(df: pd.DataFrame) -> None:
    
#     required_columns = {'sentiment', 'sentiment_label', 'clean_text'}
#     if not required_columns.issubset(df.columns):
#         raise DataProcessingError("Missing critical data columns in results")
        
#     if df['sentiment'].between(-1, 1).sum() != len(df):
#         raise DataProcessingError("Invalid sentiment values detected")


# # VISUALIZATION ENGINE
# def create_dashboard(df: pd.DataFrame, cfg: AppConfig) -> None:
    
#     with st.container():
#         col1, col2, col3 = st.columns([2, 2, 1])
#         col1.metric("Total Comments", len(df), help="Number of analyzed comments")
#         col2.metric("Avg Sentiment", f"{df['sentiment'].mean():.2f}", 
#                   delta_color="off" if abs(df['sentiment'].mean()) < 0.1 else "normal")
#         col3.download_button(
#             "📥 Export Data",
#             data=df.to_csv(index=False),
#             file_name=f"reddit_analysis_{datetime.now():%Y%m%d}.csv",
#             mime="text/csv"
#         )

#     with st.expander("📈 Advanced Analytics", expanded=True):
#         tab1, tab2 = st.tabs(["Sentiment Distribution", "Word Cloud"])
        
#         with tab1:
#             fig = px.histogram(df, x="sentiment", nbins=50, 
#                              color_discrete_sequence=[cfg.color_scheme["Positive"]])
#             fig.update_layout(template="plotly_dark", bargap=0.1)
#             st.plotly_chart(fig, use_container_width=True)

#         with tab2:
#             generate_word_cloud(df, cfg)

# def generate_word_cloud(df: pd.DataFrame, cfg: AppConfig) -> None:
    
#     col1, col2 = st.columns([3, 1])
#     with col2:
#         max_words = st.slider("Max Words", 50, 500, cfg.wordcloud_max_words)
#         colormap = st.selectbox("Color Theme", ["viridis", "plasma", "inferno"])
    
#     text = " ".join(df["clean_text"].sample(min(1000, len(df)), replace=True))
#     wc = WordCloud(
#         width=1200,
#         height=600,
#         background_color='rgba(255, 255, 255, 0)',
#         colormap=colormap,
#         max_words=max_words
#     ).generate(text)
    
#     with col1:
#         st.image(wc.to_array(), use_container_width=True)


# # MAIN APPLICATION
# def main() -> None:
    
#     cfg = AppConfig()
#     configure_ui()
    
#     st.title("📊 Reddit Sentiment Analytics Pro")
#     initialize_session_state()
    
#     with st.sidebar:
#         render_controls(cfg)
#         render_credits()

#     if st.session_state.get("run_analysis"):
#         execute_analysis_flow(cfg)

#     if 'results' in st.session_state:
#         create_dashboard(st.session_state.results, cfg)

# def initialize_session_state() -> None:
    
#     if 'analysis_history' not in st.session_state:
#         st.session_state.analysis_history = []
#     if 'run_analysis' not in st.session_state:
#         st.session_state.run_analysis = False

# def render_controls(cfg: AppConfig) -> None:
    
#     with st.expander("⚙️ Control Panel", expanded=True):
#         st.text_input("Search Topic", key="keyword", value="artificial intelligence",
#                     help="Enter any topic or keyword combination")
#         st.slider("Analysis Depth", 50, 1000, 200, 50, key="post_limit",
#                 help="Number of posts to analyze (higher = more accurate)")
        
#         if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
#             st.session_state.run_analysis = True
            
#         if st.button("🧹 Clear Cache", use_container_width=True):
#             st.cache_data.clear()
#             st.session_state.clear()

# def render_credits() -> None:
    
#     current_year = datetime.now().year
#     st.markdown("---")
#     st.markdown(f"""
#     <div style="text-align: center; padding: 1rem;">
#         <h4>Developed by Sachin Mallah</h4>
#         <p style="margin: 1rem 0;">
#             <a href="https://github.com/SachinMallah" target="_blank" 
#                style="color: #00CC96; text-decoration: none; margin: 0 1rem;">
#                 <i class="fab fa-github"></i> GitHub
#             </a>
#             <a href="www.linkedin.com/in/sachin-mallah" target="_blank" 
#                style="color: #00CC96; text-decoration: none; margin: 0 1rem;">
#                 <i class="fab fa-linkedin"></i> LinkedIn
#             </a>
#         </p>
#         <p style="font-size: 0.8rem; color: #636EFA;">
#             Reddit Sentiment Analytics Pro <br>
#             {current_year}
#         </p>
#     </div>
#     """, unsafe_allow_html=True)

# def execute_analysis_flow(cfg: AppConfig) -> None:
    
#     with st.spinner("Quantum analysis in progress..."):
#         start_time = time.time()
#         try:
#             results_df = analyze_comments(
#                 cfg,
#                 st.session_state.keyword,
#                 st.session_state.post_limit
#             )
            
#             if not results_df.empty:
#                 st.session_state.results = results_df
#                 st.session_state.analysis_history.append({
#                     "timestamp": datetime.now(),
#                     "keyword": st.session_state.keyword,
#                     "results": results_df
#                 })
#                 st.success(f"Analyzed {len(results_df)} comments in {time.time()-start_time:.2f}s")
#                 st.balloons()
#             else:
#                 st.warning("No data found for this query")
                
#         except Exception as e:
#             handle_error(e, "Main processing loop")
#         finally:
#             st.session_state.run_analysis = False

# if __name__ == "__main__":
#     main()





























# this code in integrated with reddit api as hugging face want
from __future__ import annotations
import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import praw
import re
import time
import nltk
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from nltk.sentiment import SentimentIntensityAnalyzer
from dataclasses import dataclass, field
from wordcloud import WordCloud
from loguru import logger
from dotenv import load_dotenv
from time import sleep
import logging


# CONFIGURATION
@dataclass(frozen=True)
class AppConfig:
    min_comment_length: int = 15
    max_comment_length: int = 512
    vader_workers: int = os.cpu_count() * 4
    color_scheme: Dict[str, str] = field(
        default_factory=lambda: {"Negative": "#EF553B", "Neutral": "#636EFA", "Positive": "#00CC96"}
    )
    safety_margin: float = 0.2
    wordcloud_max_words: int = 200
    sentiment_bins: Tuple[float] = (-1.0, -0.25, 0.25, 1.0)

# REDDIT CLIENT IMPLEMENTATION
class RedditClient:
    def __init__(self):
        self._verify_credentials()
        self.reddit = self._initialize_reddit()
        
    def _verify_credentials(self):
        """Validate Reddit API credentials from Streamlit secrets"""
        required = ["REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT"]
        missing = [cred for cred in required if cred not in st.secrets]
        if missing:
            handle_error(
                ValueError(f"Missing credentials: {', '.join(missing)}"),
                "Reddit API Configuration"
            )
            
    def _initialize_reddit(self) -> praw.Reddit:
        """Create authenticated Reddit instance"""
        try:
            reddit = praw.Reddit(
                client_id=st.secrets["REDDIT_CLIENT_ID"],
                client_secret=st.secrets["REDDIT_CLIENT_SECRET"],
                user_agent=st.secrets["REDDIT_USER_AGENT"]
            )
            if not reddit.user.me():
                raise ConnectionError("Reddit API authentication failed")
            return reddit
        except Exception as e:
            handle_error(e, "Reddit API Initialization")
            
    def fetch_comments(self, keyword: str, limit: int = 200) -> pd.DataFrame:
        """Fetch comments from Reddit API"""
        try:
            comments = []
            submissions = self.reddit.subreddit("all").search(
                query=keyword,
                limit=limit,
                time_filter="month"
            )
            
            for submission in submissions:
                submission.comments.replace_more(limit=0)
                for comment in submission.comments.list():
                    comments.append({
                        "author": str(comment.author),
                        "body": comment.body,
                        "score": comment.score,
                        "created_utc": comment.created_utc,
                        "subreddit": str(comment.subreddit)
                    })
                    
            return pd.DataFrame(comments)
            
        except praw.exceptions.PRAWException as e:
            handle_error(e, f"Reddit API Error: {keyword}")
            return pd.DataFrame()

# PERFORMANCE OPTIMIZATIONS
sys.path.append(str(Path(__file__).parent.parent))
nltk.download('vader_lexicon', quiet=True)

# ENHANCED STREAMLIT UI (unchanged)
def configure_ui() -> None:
    st.set_page_config(
        page_title="Reddit Analytics Pro",
        layout="wide",
        page_icon="📊",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/SachinMallah/Reddit-Sentiment-Analysis',
            'Report a bug': "mailto: Sachinmallah118@gmail.com",
            'About': "### Reddit Sentiment Analytics"
        }
    )
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    .stApp { 
        font-family: 'Inter', sans-serif; 
        background: #0f0f0f; 
        color: #ffffff;
    }
    .st-emotion-cache-1dp5vir { 
        background: linear-gradient(90deg, #00cc96 0%, #636efa 100%);
    }
    .element:hover { 
        transform: translateY(-3px); 
        transition: all 0.3s ease;
    }
    @media (max-width: 768px) { 
        .responsive-grid { 
            grid-template-columns: 1fr !important; 
        }
    }
    </style>
    """, unsafe_allow_html=True)

# ERROR HANDLING SYSTEM (unchanged)
class AnalysisError(Exception):
    pass

class DataProcessingError(AnalysisError):
    pass

def handle_error(e: Exception, context: str = "") -> None:
    error_id = f"ERR-{int(time.time())}"
    logger.error(f"{error_id} - {context}: {str(e)}")
    error_template = f"""
    <div style="border-left: 4px solid #EF553B; padding: 1rem; background: #1a1a1a;">
        <h3 style="color: #EF553B;">Error {error_id}</h3>
        <p><strong>Context:</strong> {context}</p>
        <p><strong>Details:</strong> {str(e)}</p>
        <p><small>{datetime.now().isoformat()}</small></p>
    </div>
    """
    st.markdown(error_template, unsafe_allow_html=True)
    st.session_state.last_error = error_id
    st.stop()

# PARALLEL PROCESSING ENGINE (unchanged)
def parallel_vader(texts: List[str], cfg: AppConfig) -> List[float]:
    analyzer = SentimentIntensityAnalyzer()
    validated_texts = [str(t)[:cfg.max_comment_length] if isinstance(t, (str, bytes)) else '' for t in texts]
    
    with ThreadPoolExecutor(max_workers=cfg.vader_workers) as executor:
        futures = {executor.submit(analyzer.polarity_scores, t): i for i, t in enumerate(validated_texts)}
        results = [0.0] * len(texts)
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()["compound"]
            except Exception as e:
                logger.warning(f"VADER processing failed for index {idx}: {str(e)}")
                results[idx] = 0.0
        return results

def vectorized_clean(text_series: pd.Series, cfg: AppConfig) -> pd.Series:
    return (
        text_series.astype(str)
        .str[:cfg.max_comment_length]
        .str.lower()
        .str.replace(r'[^\w\s]', '', regex=True)
        .str.strip()
        .fillna('')
    )

# CORE PROCESSING PIPELINE (unchanged)
@st.cache_data(ttl=300, show_spinner=False)
def analyze_comments(cfg: AppConfig, keyword: str, post_limit: int) -> pd.DataFrame:
    try:
        with st.spinner("🌐 Fetching Reddit Data..."):
            client = RedditClient()
            df = client.fetch_comments(keyword, post_limit)
            if df.empty:
                return pd.DataFrame()

        df["clean_text"] = vectorized_clean(df["body"], cfg)
        valid_comments = df[df["clean_text"].str.len() >= cfg.min_comment_length]
        
        if valid_comments.empty:
            return pd.DataFrame()

        with st.spinner("🧠 Analyzing Sentiment..."):
            scores = parallel_vader(valid_comments["clean_text"].tolist(), cfg)
            valid_comments = valid_comments.assign(sentiment=scores)

        valid_comments["sentiment_label"] = pd.cut(
            valid_comments["sentiment"],
            bins=cfg.sentiment_bins,
            labels=["Negative", "Neutral", "Positive"]
        ).fillna("Neutral")
        
        validate_results(valid_comments)
        return valid_comments

    except Exception as e:
        handle_error(e, "Core processing pipeline")

def validate_results(df: pd.DataFrame) -> None:
    required_columns = {'sentiment', 'sentiment_label', 'clean_text'}
    if not required_columns.issubset(df.columns):
        raise DataProcessingError("Missing critical data columns in results")
    if df['sentiment'].between(-1, 1).sum() != len(df):
        raise DataProcessingError("Invalid sentiment values detected")

# VISUALIZATION ENGINE (unchanged)
def create_dashboard(df: pd.DataFrame, cfg: AppConfig) -> None:
    with st.container():
        col1, col2, col3 = st.columns([2, 2, 1])
        col1.metric("Total Comments", len(df), help="Number of analyzed comments")
        col2.metric("Avg Sentiment", f"{df['sentiment'].mean():.2f}", 
                  delta_color="off" if abs(df['sentiment'].mean()) < 0.1 else "normal")
        col3.download_button(
            "📥 Export Data",
            data=df.to_csv(index=False),
            file_name=f"reddit_analysis_{datetime.now():%Y%m%d}.csv",
            mime="text/csv"
        )

    with st.expander("📈 Advanced Analytics", expanded=True):
        tab1, tab2 = st.tabs(["Sentiment Distribution", "Word Cloud"])
        
        with tab1:
            fig = px.histogram(df, x="sentiment", nbins=50, 
                             color_discrete_sequence=[cfg.color_scheme["Positive"]])
            fig.update_layout(template="plotly_dark", bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            generate_word_cloud(df, cfg)

def generate_word_cloud(df: pd.DataFrame, cfg: AppConfig) -> None:
    col1, col2 = st.columns([3, 1])
    with col2:
        max_words = st.slider("Max Words", 50, 500, cfg.wordcloud_max_words)
        colormap = st.selectbox("Color Theme", ["viridis", "plasma", "inferno"])
    
    text = " ".join(df["clean_text"].sample(min(1000, len(df)), replace=True))
    wc = WordCloud(
        width=1200,
        height=600,
        background_color='rgba(255, 255, 255, 0)',
        colormap=colormap,
        max_words=max_words
    ).generate(text)
    
    with col1:
        st.image(wc.to_array(), use_container_width=True)

# MAIN APPLICATION (unchanged)
def main() -> None:
    cfg = AppConfig()
    configure_ui()
    st.title("📊 Reddit Sentiment Analytics Pro")
    initialize_session_state()
    
    with st.sidebar:
        render_controls(cfg)
        render_credits()

    if st.session_state.get("run_analysis"):
        execute_analysis_flow(cfg)

    if 'results' in st.session_state:
        create_dashboard(st.session_state.results, cfg)

def initialize_session_state() -> None:
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    if 'run_analysis' not in st.session_state:
        st.session_state.run_analysis = False

def render_controls(cfg: AppConfig) -> None:
    with st.expander("⚙️ Control Panel", expanded=True):
        st.text_input("Search Topic", key="keyword", value="artificial intelligence",
                    help="Enter any topic or keyword combination")
        st.slider("Analysis Depth", 50, 1000, 200, 50, key="post_limit",
                help="Number of posts to analyze (higher = more accurate)")
        
        if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
            st.session_state.run_analysis = True
            
        if st.button("🧹 Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.session_state.clear()

def render_credits() -> None:
    current_year = datetime.now().year
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; padding: 1rem;">
        <h4>Developed by Sachin Mallah</h4>
        <p style="margin: 1rem 0;">
            <a href="https://github.com/SachinMallah" target="_blank" 
               style="color: #00CC96; text-decoration: none; margin: 0 1rem;">
                <i class="fab fa-github"></i> GitHub
            </a>
            <a href="www.linkedin.com/in/sachin-mallah" target="_blank" 
               style="color: #00CC96; text-decoration: none; margin: 0 1rem;">
                <i class="fab fa-linkedin"></i> LinkedIn
            </a>
        </p>
        <p style="font-size: 0.8rem; color: #636EFA;">
            Reddit Sentiment Analytics Pro <br>
            {current_year}
        </p>
    </div>
    """, unsafe_allow_html=True)

def execute_analysis_flow(cfg: AppConfig) -> None:
    with st.spinner("Quantum analysis in progress..."):
        start_time = time.time()
        try:
            results_df = analyze_comments(
                cfg,
                st.session_state.keyword,
                st.session_state.post_limit
            )
            
            if not results_df.empty:
                st.session_state.results = results_df
                st.session_state.analysis_history.append({
                    "timestamp": datetime.now(),
                    "keyword": st.session_state.keyword,
                    "results": results_df
                })
                st.success(f"Analyzed {len(results_df)} comments in {time.time()-start_time:.2f}s")
                st.balloons()
            else:
                st.warning("No data found for this query")
                
        except Exception as e:
            handle_error(e, "Main processing loop")
        finally:
            st.session_state.run_analysis = False

if __name__ == "__main__":
    main()
