import os
import time
import json
import logging
import requests
import re
from threading import Thread, Lock
from datetime import datetime, timedelta
from dotenv import load_dotenv
from time_utils import now_ist

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class PerplexitySentimentService:
    def __init__(self, interval_minutes=10):
        self.api_key = os.getenv("PERPLEXITY_API_KEY")
        self.url = "https://api.perplexity.ai/chat/completions"
        self.interval = interval_minutes * 60
        self.last_score = 0.0
        self.last_summary = "Initializing..."
        self.last_drivers = []
        self.last_update_time = None
        self.lock = Lock()
        self.running = False
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _clean_json_response(self, content):
        """Remove Markdown code blocks if Perplexity adds them."""
        # Regex to find json content between ```json and ``` or just brackets
        # Matches content starting with { and ending with } possibly surrounded by markdown blocks
        # First try to extract from code blocks
        code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1)
            
        # Fallback: try to find the first valid JSON object
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json_match.group(0)
            
        return content

    def _build_payload(self):
        """Construct the payload with dynamic time-based prompt."""
        current_time = now_ist()
        prev_time = current_time - timedelta(minutes=60)
        
        now_str = current_time.strftime("%Y-%m-%d %H:%M IST")
        prev_hour_str = prev_time.strftime("%H:%M IST")
        
        system_prompt = (
            "You are a senior quantitative trading analyst for the Indian Stock Market (NIFTY 50 and BANK NIFTY). "
            "Your job is to analyze real-time news from the last 60 minutes and quantify the market sentiment. "
            "Output strict JSON only."
        )
        
        user_prompt = (
            f"Current Time (IST): {now_str}. "
            f"Search for news published ONLY between {prev_hour_str} and {now_str}. "
            "Ignore news older than 2 hours. "
            "Prioritize sources: MoneyControl, Economic Times, LiveMint, Reuters India, ZEE BUSINESS, CNBCTV 18, NDTV Profit, ET NOW and major Twitter financial handles. "
            "\n\nFocus on:\n"
            "1. FII/DII activity or rumors.\n"
            "2. Unexpected earnings results (HDFCBANK, Reliance, ICICI, SBI, KOTAK BANK,INFOSYS, WIPRO,ITC, etc.).\n"
            "3. Global cues (US Futures, SGX Nifty/GIFT Nifty, Crude Oil prices).\n"
            "4. Macro announcements (RBI, Inflation, GDP).\n"
            "\n"
            "Based on the search results, calculate a Sentiment Score between -1.0 (Extreme Bearish) and +1.0 (Extreme Bullish).\n"
            "0.0 is Neutral.\n"
            "\n"
            "Rules for Scoring:\n"
            "- > 0.5: Breakout news, Rate cuts, Massive FII buying.\n"
            "- 0.1 to 0.4: General positive trend, good earnings.\n"
            "- -0.1 to 0.1: Mixed signals, consolidation, no major news.\n"
            "- -0.4 to -0.1: Profit booking, weak global cues, rising crude.\n"
            "- < -0.5: War, Pandemic, Surprise Rate Hike, Crash.\n"
            "\n"
            "Output STRICT JSON format ONLY. Do not output markdown.\n"
            "Structure:\n"
            "{\n"
            "  \"score\": <float>,\n"
            "  \"confidence\": <float 0.0 to 1.0>,\n"
            "  \"reasoning\": \"<short 1-sentence summary>\",\n"
            "  \"key_drivers\": [\"<driver1>\", \"<driver2>\"]\n"
            "}"
        )
        
        return {
            "model": "sonar-pro",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1
        }

    def fetch_sentiment(self):
        if not self.api_key:
            logger.error("Perplexity API Key missing! Please set PERPLEXITY_API_KEY in your .env file.")
            return None

        try:
            logger.info("Querying Perplexity API for market sentiment...")
            start_time = time.time()
            
            # Generate payload with fresh timestamp
            payload = self._build_payload()
            
            response = requests.post(self.url, json=payload, headers=self.headers, timeout=60)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                raw_content = data['choices'][0]['message']['content']
                
                # Parse JSON
                clean_content = self._clean_json_response(raw_content)
                result = json.loads(clean_content)
                
                score = float(result.get('score', 0.0))
                confidence = float(result.get('confidence', 0.0))
                reasoning = result.get('reasoning', 'No reasoning provided')
                drivers = result.get('key_drivers', [])
                
                with self.lock:
                    self.last_score = score
                    self.last_summary = reasoning
                    self.last_drivers = drivers
                    self.last_update_time = now_ist()
                
                logger.info(f"[PERPLEXITY] Success ({duration:.2f}s). Score: {score}, Summary: {reasoning}")
                
                return {
                    'score': score,
                    'summary': reasoning,
                    'drivers': drivers,
                    'confidence': confidence
                }
            else:
                logger.error(f"[PERPLEXITY] API Error {response.status_code}: {response.text}")
                return None

        except json.JSONDecodeError as e:
            logger.error(f"[PERPLEXITY] JSON Decode Error: {e}")
            return None
        except Exception as e:
            logger.error(f"[PERPLEXITY] Exception: {e}")
            return None

    def start_background_loop(self):
        """Runs the fetcher in a background thread."""
        if not self.api_key:
            logger.warning("Perplexity Service disabled (No API Key)")
            return

        self.running = True
        def loop():
            logger.info(f"Perplexity Service started (Interval: {self.interval}s)")
            while self.running:
                # Initial fetch
                self.fetch_sentiment()
                
                # Sleep in chunks to allow faster shutdown
                for _ in range(int(self.interval)):
                    if not self.running:
                        break
                    time.sleep(1)
        
        t = Thread(target=loop, daemon=True, name="PerplexityWorker")
        t.start()

    def get_current_sentiment(self):
        """Return the latest sentiment data in a safe structure."""
        with self.lock:
            return {
                'score': self.last_score,
                'summary': self.last_summary,
                'drivers': self.last_drivers,
                'updated_at': self.last_update_time
            }

    def stop(self):
        self.running = False

