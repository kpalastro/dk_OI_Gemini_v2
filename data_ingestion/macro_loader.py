import logging
import requests
import csv
import io
import re
from datetime import datetime, timedelta
from typing import Dict, Optional, Any

# Constants
NSE_FII_JSON_URL = "https://www.nseindia.com/api/fiidiiTradeReact"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://www.nseindia.com/reports/fii-dii"
}

def _fetch_fii_dii_nse_json(session) -> Optional[Dict[str, float]]:
    """Try fetching from NSE API (JSON)."""
    try:
        # NSE requires a visit to the homepage or report page first to set cookies
        session.get("https://www.nseindia.com/reports/fii-dii", headers=HEADERS, timeout=5)
        
        response = session.get(NSE_FII_JSON_URL, headers=HEADERS, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            fii_net = 0.0
            dii_net = 0.0
            
            # Output format: [{"category":"DII **", "netValue":"824.46"}, ...]
            for item in data:
                category = item.get('category', '')
                try:
                    net_val = float(item.get('netValue', 0))
                    if "FII/FPI" in category:
                        fii_net = net_val
                    elif "DII" in category:
                        dii_net = net_val
                except (ValueError, TypeError):
                    continue
            
            return {
                'fii_net': fii_net,
                'dii_net': dii_net,
                'fii_dii_net': fii_net + dii_net
            }
    except Exception as e:
        logging.debug(f"NSE API JSON fetch failed: {e}")
    return None

def fetch_fii_dii_data() -> Dict[str, float]:
    """
    Fetch FII/DII provisional data.
    Strategy: Try NSE JSON API.
    """
    try:
        session = requests.Session()
        
        # Attempt 1: NSE JSON API
        data = _fetch_fii_dii_nse_json(session)
        if data:
            logging.info(f"✓ Macro Data (NSE API): FII Net: {data['fii_net']} Cr, DII Net: {data['dii_net']} Cr")
            return data
            
        logging.warning("⚠ Could not fetch FII/DII data. Using neutral bias.")
        return {'fii_net': 0.0, 'dii_net': 0.0, 'fii_dii_net': 0.0}

    except Exception as e:
        logging.error(f"FII/DII fetch error: {e}")
        return {'fii_net': 0.0, 'dii_net': 0.0, 'fii_dii_net': 0.0}

def find_macro_tokens(kite_client) -> Dict[str, int]:
    """
    Find instrument tokens for active USDINR and CRUDEOIL futures.
    Logic:
    - USDINR (CDS): Exclude weekly contracts (regex check). Pick NEXT MONTH expiry (Month 2).
    - CRUDEOIL (MCX): Pick NEXT MONTHLY expiry (Front Month).
    """
    macro_tokens = {}
    try:
        # 1. USDINR (NSE-CDS)
        try:
            instruments_cds = kite_client.instruments('CDS')
            # Filter for USDINR Futures
            usdinr_futs = [
                i for i in instruments_cds 
                if i['name'] == 'USDINR' and i['instrument_type'] == 'FUT'
            ]
            
            # Filter out weekly contracts
            # Monthly symbols: "USDINR25NOVFUT" (Standard format)
            # Weekly symbols: "USDINR25N21FUT", "USDINR25D05FUT" (Contain specific dates/week codes)
            monthly_futs = []
            for i in usdinr_futs:
                symbol = i['tradingsymbol']
                # Regex: Look for MMMFUT pattern (e.g., NOVFUT, DECFUT) avoiding digits in the month part
                if re.search(r'[A-Z]{3}FUT$', symbol):
                    monthly_futs.append(i)
            
            if monthly_futs:
                monthly_futs.sort(key=lambda x: x['expiry'])
                
                # User rule: "Always use next month expiry" for USDINR
                # Index 0 = Current Month (e.g., Nov), Index 1 = Next Month (e.g., Dec)
                if len(monthly_futs) > 1:
                    target = monthly_futs[1]  # Next Month
                    desc = "Next Month"
                else:
                    target = monthly_futs[0]  # Fallback to Current
                    desc = "Current Month"
                
                macro_tokens['USDINR'] = target['instrument_token']
                logging.info(f"✓ Found USDINR Future ({desc}): {target['tradingsymbol']} ({target['instrument_token']})")
            else:
                logging.debug("No monthly USDINR Futures found.")
                
        except Exception as e:
            logging.debug(f"USDINR search failed: {e}")

        # 2. CRUDEOIL (MCX)
        try:
            instruments_mcx = kite_client.instruments('MCX')
            crude_futs = [
                i for i in instruments_mcx 
                if i['name'] == 'CRUDEOIL' and i['instrument_type'] == 'FUT'
            ]
            if crude_futs:
                crude_futs.sort(key=lambda x: x['expiry'])
                
                # User rule: "Always use next Monthly expiry" 
                # In commodities, "Next Monthly" typically means the Front Month (most active)
                # unless it's very close to expiry. Assuming Front Month (Index 0) here.
                target = crude_futs[0]
                
                macro_tokens['CRUDEOIL'] = target['instrument_token']
                logging.info(f"✓ Found CRUDEOIL Future (Front): {target['tradingsymbol']} ({target['instrument_token']})")
            else:
                logging.debug("CRUDEOIL Futures not found.")
        except Exception as e:
            logging.debug(f"CRUDEOIL search failed: {e}")

    except Exception as e:
        logging.error(f"Error finding macro tokens: {e}")
    
    return macro_tokens
