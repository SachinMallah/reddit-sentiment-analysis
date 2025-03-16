"""
Reddit API client with error handling and rate limiting
"""
import praw
import pandas as pd
from dotenv import load_dotenv
import os
from datetime import datetime
from time import sleep
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RedditClient:
    def __init__(self):
        """Initialize Reddit API client"""
        load_dotenv()
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT"),
            timeout=10  # 10 seconds timeout
        )
        self._validate_credentials()
    
    def _validate_credentials(self):
        """Verify API credentials are working"""
        try:
            self.reddit.user.me()
            logger.info("✅ Successfully connected to Reddit API")
        except Exception as e:
            logger.error(f"❌ Reddit API connection failed: {str(e)}")
            raise
    
    def fetch_comments(self, keyword, limit=100, max_retries=3):
        """
        Fetch comments from Reddit with error handling
        Args:
            keyword: Search term
            limit: Number of posts to process
            max_retries: Number of retry attempts
        Returns:
            DataFrame with comments
        """
        comments = []
        retries = 0
        
        while retries < max_retries:
            try:
                posts = self.reddit.subreddit("all").search(
                    query=keyword,
                    limit=limit,
                    time_filter="month",
                    syntax="lucene"
                )
                
                for post in posts:
                    try:
                        post.comments.replace_more(limit=0)
                        for comment in post.comments.list():
                            comments.append({
                                "id": comment.id,
                                "body": comment.body,
                                "score": comment.score,
                                "created": datetime.fromtimestamp(comment.created_utc),
                                "author": str(comment.author),
                                "post_title": post.title
                            })
                            if len(comments) >= limit * 10:  # Safety break
                                break
                    except Exception as post_error:
                        logger.warning(f"⚠️ Error processing post: {str(post_error)}")
                
                return pd.DataFrame(comments)
            
            except Exception as e:
                retries += 1
                logger.warning(f"⚠️ Attempt {retries}/{max_retries} failed: {str(e)}")
                sleep(2 ** retries)  # Exponential backoff
        
        logger.error("❌ Maximum retries exceeded")
        return pd.DataFrame()
    
