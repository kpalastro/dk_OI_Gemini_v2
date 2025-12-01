import os
import time
import json
import logging
import requests
import re
from threading import Thread, Lock
from datetime import datetime, timezone
from zoneinfo import ZoneInfo  # Python 3.9+
from dotenv import load_dotenv


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
        self.lock = Lock()
        self.running = False

        # Static parts of the payload
        self.base_system_prompt = (
            "You are a senior quantitative trading analyst for the Indian Stock Market "
            "(NIFTY 50 and BANK NIFTY). "
            "Your job is to analyze real-time news and quantify the market sentiment. "
            "Output strict JSON only."
        )

        self.temperature = 0.1

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _build_user_prompt(self) -> str:
        """
        Build a fresh, time-aware user prompt each call.
        Uses current IST time and enforces a strict last-60-minutes window.
        """
        now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
        now_str = now_ist.strftime("%Y-%m-%d %H:%M IST")

        prompt = (
            f"Current time is {now_str}. "
            "Search for the latest breaking financial news affecting the Indian Stock Market. "
            "Strongly prefer news published in the LAST 60 MINUTES relative to this time. "
            "If no valid news is found within the last 60 minutes, fall back to the latest same-day news "
            "or last-close news, but clearly label the time range in the output.\n\n"
            "Prioritize sources: MoneyControl, Economic Times, LiveMint, Reuters India, "
            "ZEE BUSINESS, CNBCTV18, NDTV Profit, ET NOW and major Twitter financial handles.\n\n"
            "Focus on:\n"
            "1. FII/DII activity or rumors.\n"
            "2. Unexpected earnings results (HDFCBANK, Reliance, ICICI, SBI, KOTAK BANK, "
            "INFOSYS, WIPRO, ITC, etc.).\n"
            "3. Global cues (US Futures, SGX Nifty/GIFT Nifty, Crude Oil prices).\n"
            "4. Macro announcements (RBI, Inflation, GDP).\n\n"
            "Based on the search results, calculate a Sentiment Score between -1.0 (Extreme Bearish) "
            "and +1.0 (Extreme Bullish). 0.0 is Neutral.\n\n"
            "Rules for Scoring:\n"
            "- > 0.5: Breakout news, Rate cuts, Massive FII buying.\n"
            "- 0.1 to 0.4: General positive trend, good earnings.\n"
            "- -0.1 to 0.1: Mixed signals, consolidation, no major news.\n"
            "- -0.4 to -0.1: Profit booking, weak global cues, rising crude.\n"
            "- < -0.5: War, Pandemic, Surprise Rate Hike, Crash.\n\n"
            "Output STRICT JSON format ONLY. Do not output markdown.\n"
            "Structure:\n"
            "{\n"
            "  \"score\": <float>,\n"
            "  \"confidence\": <float 0.0 to 1.0>,\n"
            "  \"reasoning\": \"<short 1-sentence summary>\",\n"
            "  \"key_drivers\": [\"<driver1>\", \"<driver2>\"],\n"
            "  \"news_window_start\": \"<timestamp of oldest news used>\",\n"
            "  \"news_window_end\": \"<timestamp of newest news used>\",\n"
            "  \"data_staleness_minutes\": <int approx minutes since newest news>,\n"
            "  \"has_strict_60m_news\": <bool true if valid news found in last 60m>\n"
            "}"
        )
        return prompt

    def _build_payload(self) -> dict:
        """
        Build a fresh payload dict for every request
        to prevent reuse of stale messages.[web:19]
        """
        payload = {
            "model": "sonar-pro",
            "messages": [
                {
                    "role": "system",
                    "content": self.base_system_prompt
                },
                {
                    "role": "user",
                    "content": self._build_user_prompt()
                }
            ],
            "temperature": self.temperature
        }
        return payload

    def _clean_json_response(self, content):
        """Remove Markdown code blocks if Perplexity adds them."""
        # Matches content starting with { and ending with } possibly surrounded by markdown blocks
        code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1)

        # Fallback: try to find the first valid JSON object
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json_match.group(0)

        return content

    def fetch_sentiment(self):
        if not self.api_key:
            logger.error("Perplexity API Key missing! Please set PERPLEXITY_API_KEY in your .env file.")
            return None

        try:
            logger.info("Querying Perplexity API for market sentiment...")
            start_time = time.time()

            payload = self._build_payload()
            response = requests.post(self.url, json=payload, headers=self.headers, timeout=60)
            duration = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                raw_content = data['choices'][0]['message']['content']
                logger.debug(f"Raw API Response: {raw_content}")

                clean_content = self._clean_json_response(raw_content)
                result = json.loads(clean_content)

                score = float(result.get('score', 0.0))
                confidence = float(result.get('confidence', 0.0))
                reasoning = result.get('reasoning', 'No reasoning provided')
                drivers = result.get('key_drivers', [])
                
                # New fields for staleness tracking
                news_window_start = result.get('news_window_start', 'N/A')
                news_window_end = result.get('news_window_end', 'N/A')
                data_staleness = result.get('data_staleness_minutes', -1)
                has_strict_60m = result.get('has_strict_60m_news', False)

                with self.lock:
                    self.last_score = score
                    self.last_summary = reasoning

                logger.info(f"[PERPLEXITY] Success ({duration:.2f}s).")
                logger.info(f"  Score: {score}")
                logger.info(f"  Confidence: {confidence}")
                logger.info(f"  Reasoning: {reasoning}")
                logger.info(f"  Drivers: {', '.join(drivers)}")
                logger.info(f"  Window: {news_window_start} to {news_window_end}")
                logger.info(f"  Staleness: {data_staleness} min | Strict 60m: {has_strict_60m}")

                return result
            else:
                logger.error(f"[PERPLEXITY] API Error {response.status_code}: {response.text}")
                return None

        except json.JSONDecodeError as e:
            logger.error(f"[PERPLEXITY] JSON Decode Error: {e}")
            logger.error(f"Content that failed to parse: {raw_content}")
            return None
        except Exception as e:
            logger.error(f"[PERPLEXITY] Exception: {e}")
            return None


if __name__ == "__main__":
    print("=" * 70)
    print("PERPLEXITY SENTIMENT ANALYZER (Standalone Test)")
    print("=" * 70)

    service = PerplexitySentimentService()

    if not service.api_key:
        print("❌ ERROR: PERPLEXITY_API_KEY not found in .env file.")
        print("Please add your API key to .env and try again.")
        exit(1)

    print("Initializing request... (This may take 10-30 seconds)")
    result = service.fetch_sentiment()

    if result:
        print("\n" + "=" * 70)
        print("✅ RESULT RECEIVED")
        print("=" * 70)
        print(json.dumps(result, indent=2))
        print("\nAnalysis Complete.")
    else:
        print("\n❌ Failed to fetch sentiment.")
