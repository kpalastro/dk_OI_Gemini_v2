import os
import time
import json
import logging
import requests
import re
from threading import Thread, Lock
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class PerplexitySentimentService:
    def __init__(self, interval_minutes=10):
        self.api_key = os.getenv("PERPLEXITY_API_KEY")
        self.url = "https://api.perplexity.ai/chat/completions"
        self.interval = interval_minutes * 60
        self.last_score = 0.0
        self.last_summary = "Initializing..."
        self.lock = Lock()
        self.running = False
        
        # This is the "End-to-End Prompt" logic
        self.payload_template = {
            "model": "sonar-pro", # Uses online search capabilities
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a quantitative trading algorithm. "
                        "Analyze real-time news for NIFTY 50 and BANKNIFTY. "
                        "Output strict JSON only."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Search current financial news (last 60 mins) for Indian Markets "
                        "(MoneyControl, Economic Times, Global Cues). "
                        "Determine a sentiment score from -1.0 (Bearish) to 1.0 (Bullish). "
                        "Return JSON format: "
                        "{ \"score\": float, \"summary\": string, \"drivers\": [string] }"
                    )
                }
            ],
            "temperature": 0.1 # Low temp for consistent, analytical outputs
        }
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _clean_json_response(self, content):
        """Remove Markdown code blocks if Perplexity adds them."""
        # Regex to find json content between ```json and ``` or just brackets
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            return match.group(0)
        return content

    def fetch_sentiment(self):
        if not self.api_key:
            logging.error("Perplexity API Key missing!")
            return 0.0

        try:
            logging.info("Querying Perplexity API for market sentiment...")
            response = requests.post(self.url, json=self.payload_template, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                raw_content = data['choices'][0]['message']['content']
                
                # Parse JSON
                clean_content = self._clean_json_response(raw_content)
                result = json.loads(clean_content)
                
                score = float(result.get('score', 0.0))
                summary = result.get('summary', 'No summary provided')
                
                with self.lock:
                    self.last_score = score
                    self.last_summary = summary
                
                logging.info(f"[PERPLEXITY] Success. Score: {score}, Summary: {summary}")
                return score
            else:
                logging.error(f"[PERPLEXITY] API Error {response.status_code}: {response.text}")
                return self.last_score

        except Exception as e:
            logging.error(f"[PERPLEXITY] Exception: {e}")
            return self.last_score

    def start_background_loop(self):
        """Runs the fetcher in a background thread."""
        self.running = True
        def loop():
            while self.running:
                # 1. Check if Market is Open (Optional optimization to save credits)
                # if not _is_market_open(): time.sleep(60); continue
                
                # 2. Fetch
                self.fetch_sentiment()
                
                # 3. Wait
                time.sleep(self.interval)
        
        t = Thread(target=loop, daemon=True, name="PerplexityWorker")
        t.start()
        logging.info(f"Perplexity Sentiment Service started (Interval: {self.interval}s)")

    def get_current_sentiment(self):
        with self.lock:
            return self.last_score