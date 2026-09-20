import os
import pandas as pd
import re
import socket
import ssl
import requests
import whois
import dns.resolver
import tldextract
from datetime import datetime
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from tqdm import tqdm
from scipy.stats import randint

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class FeatureExtractor:

 def extract_features(self, url):
    features = {}

    def is_indexed(url):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        query = f"site:{url}"
        search_url = f"https://www.google.com/search?q={query}"
    
        try:
            response = requests.get(search_url, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")
            # Google shows a message if no results
            if "did not match any documents" in soup.text:
                return -1
            else:
                return 1
        except:
            return 0
    
    def google_safe_browsing(url):
        API_KEY = os.getenv("SAFE_BROWSING_API_KEY", "")
        if not API_KEY or API_KEY == "your_google_safe_browsing_api_key_here":
            return 1  # Default to legitimate if API key is not configured

        endpoint = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={API_KEY}"

        payload = {
            "client": {"clientId": "fyp", "clientVersion": "1.0"},
            "threatInfo": {
                "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING"],
                "platformTypes": ["ANY_PLATFORM"],
                "threatEntryTypes": ["URL"],
                "threatEntries": [{"url": url}]
            }
        }

        try:
            r = requests.post(endpoint, json=payload, timeout=5)
            return -1 if "matches" in r.json() else 1
        except Exception:
            return 1

    # Helper variables
    parsed = urlparse(url)
    ext = tldextract.extract(url)
    hostname = parsed.hostname or ""
    domain = ext.registered_domain
    subdomain = ext.subdomain
    now = datetime.now()

    # 1. having_IP_Address
    ip_pattern = re.compile(r'(\d{1,3}\.){3}\d{1,3}')
    features['having_IP_Address'] = -1 if ip_pattern.match(hostname) else 1

    # 2. URL_Length
    features['URL_Length'] = 1 if len(url) < 54 else (0 if 54 <= len(url) <= 75 else -1)

    # 3. Shortening_Service
    shortening_services = ['bit.ly', 'goo.gl', 'tinyurl.com', 'ow.ly', 't.co']
    features['Shortening_Service'] = -1 if any(s in hostname for s in shortening_services) else 1

    # 4. having_At_Symbol
    features['having_At_Symbol'] = -1 if '@' in url else 1

    # 5. double_slash_redirecting
    features['double_slash_redirecting'] = -1 if url.count('//') > 1 else 1

    # 6. Prefix_Suffix
    features['Prefix_Suffix'] = -1 if '-' in domain else 1

    # 7. having_Sub_Domain
    dot_count = subdomain.count('.')
    features['having_Sub_Domain'] = 1 if dot_count == 0 else (0 if dot_count == 1 else -1)

    # 8. SSLfinal_State
    try:
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(socket.socket(), server_hostname=hostname) as s:
            s.settimeout(5)
            s.connect((hostname, 443))
            cert = s.getpeercert()
        issuer = dict(x[0] for x in cert['issuer'])
        start_date = datetime.strptime(cert['notBefore'], '%b %d %H:%M:%S %Y %Z')
        age = (now - start_date).days
        if issuer and age >= 365:
            features['SSLfinal_State'] = 1
        else:
            features['SSLfinal_State'] = -1
    except:
        features['SSLfinal_State'] = -1

    # 9. Domain_registration_length
    try:
        whois_info = whois.whois(domain)
        exp_date = whois_info.expiration_date
        if isinstance(exp_date, list):
            exp_date = exp_date[0]
        reg_days = (exp_date - now).days
        features['Domain_registration_length'] = 1 if reg_days >= 365 else -1
    except:
        features['Domain_registration_length'] = -1

    # 10. port
    features['port'] = 1 if parsed.port in [80, 443, None] else -1

    # 11. HTTPS_token
    features['HTTPS_token'] = -1 if 'https' in subdomain else 1

    # Content-based features
    try:
        resp = requests.get(url, timeout=5)
        soup = BeautifulSoup(resp.content, 'html.parser')
    except:
        soup = None

    # 12. Request_URL
    try:
        imgs = soup.find_all('img', src=True)
        total = len(imgs)
        external = sum(1 for img in imgs if domain not in img['src'])
        features['Request_URL'] = 1 if total == 0 else (1 if external/total < 0.22 else (0 if external/total <= 0.61 else -1))
    except:
        features['Request_URL'] = -1

    # 13. URL_of_Anchor
    try:
        anchors = soup.find_all('a', href=True)
        total_a = len(anchors)
        external_a = sum(1 for a in anchors if domain not in a['href'])
        features['URL_of_Anchor'] = 1 if total_a == 0 else (1 if external_a/total_a < 0.31 else (0 if external_a/total_a <= 0.67 else -1))
    except:
        features['URL_of_Anchor'] = -1

    # 14. Links_in_tags
    try:
        metas = soup.find_all('meta')
        links = soup.find_all('link', href=True)
        scripts = soup.find_all('script', src=True)
        total_tags = len(metas) + len(links) + len(scripts)
        external_tags = sum(1 for tag in links+scripts if domain not in (tag.get('href') or tag.get('src', '')))
        features['Links_in_tags'] = 1 if total_tags == 0 else (1 if external_tags/total_tags < 0.17 else (0 if external_tags/total_tags <= 0.81 else -1))
    except:
        features['Links_in_tags'] = -1

    # 15. SFH
    try:
        forms = soup.find_all('form', action=True)
        features['SFH'] = 1 if not forms else (
            1 if all(domain in f['action'] or f['action'] == '' for f in forms) else (
            -1 if any('http' in f['action'] and domain not in f['action'] for f in forms) else 0))
    except:
        features['SFH'] = -1

    # 16. Abnormal_URL
    features['Abnormal_URL'] = -1 if domain not in url else 1

    # 17. Redirect
    try:
        features['Redirect'] = -1 if len(resp.history) > 2 else 1
    except:
        features['Redirect'] = -1

    # 18. on_mouseover
    try:
        features['on_mouseover'] = -1 if "onmouseover" in resp.text.lower() else 1
    except:
        features['on_mouseover'] = 1

    # 19. RightClick
    try:
        features['RightClick'] = -1 if "event.button==2" in resp.text.lower() else 1
    except:
        features['RightClick'] = 1

    # 20. popUpWidnow
    try:
        features['popUpWidnow'] = -1 if "alert(" in resp.text.lower() else 1
    except:
        features['popUpWidnow'] = 1

    # 21. Iframe
    try:
        features['Iframe'] = -1 if "<iframe" in resp.text.lower() else 1
    except:
        features['Iframe'] = 1

    # 22. age_of_domain
    try:
        creation_date = whois_info.creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        age_days = (now - creation_date).days
        features['age_of_domain'] = 1 if age_days >= 180 else -1
    except:
        features['age_of_domain'] = -1

    # 23. DNSRecord
    try:
        dns.resolver.resolve(domain, 'A')
        features['DNSRecord'] = 1
    except:
        features['DNSRecord'] = -1
        
    # 24. Statistical_report (placeholder)
    features['Statistical_report'] = google_safe_browsing(url)  # Replace with blacklist check

    return features

# def extract_features_from_dataframe(df, url_column="URL",type_column="type"):

    # Ensure the column exists
    # if url_column not in df.columns:
       # raise ValueError(f"Column '{url_column}' not found in DataFrame")

    # feature_list = []
    # for url in tqdm(df[url_column], desc="Extracting features"):
        # try:
            # feats = extract_features(url) 
            # feats['url'] = url  # keep original URL for traceability
        # except Exception as e:
            # If extraction fails, return -1 for all features
            # feats = {f: -1 for f in [
                # 'having_IP_Address','URL_Length','Shortening_Service','having_At_Symbol',
                # 'double_slash_redirecting','Prefix_Suffix','having_Sub_Domain','SSLfinal_State',
                # 'Domain_registration_length','port','HTTPS_token','Request_URL','URL_of_Anchor',
                # 'Links_in_tags','SFH','Submitting_to_email','Abnormal_URL','Redirect',
                # 'on_mouseover','RightClick','popUpWidnow','Iframe','age_of_domain','DNSRecord',
                # 'web_traffic','Page_Rank','Google_Index','Links_pointing_to_page','Statistical_report'
           # ]}
            # feats['url'] = url
        # feature_list.append(feats)

    # feature_df = pd.DataFrame(feature_list)

    # Merge with original DataFrame (align by index)
    # merged_df = pd.concat([df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)

    # return merged_df