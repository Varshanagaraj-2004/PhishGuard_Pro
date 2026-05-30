"""
Feature Extractor for Phishing Detection
Extracts structured features from URLs, emails, and messages.
"""

import re
import urllib.parse
from typing import List, Dict, Any


# ─────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────

SUSPICIOUS_KEYWORDS = [
    "login", "verify", "secure", "account", "update", "confirm",
    "banking", "paypal", "amazon", "apple", "microsoft", "google",
    "password", "credential", "click", "urgent", "free", "winner",
    "prize", "congratulations", "suspended", "validate", "ebay",
]

SUSPICIOUS_TLDS = [
    ".xyz", ".top", ".club", ".online", ".site", ".work", ".bid",
    ".loan", ".gq", ".ml", ".cf", ".ga", ".tk",
]

SHORTENERS = [
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "buff.ly",
    "short.io", "rebrand.ly", "is.gd", "cutt.ly",
]


# ─────────────────────────────────────────────────────────────
# URL Feature Extractor
# ─────────────────────────────────────────────────────────────

class URLFeatureExtractor:
    """Extracts 30 hand-crafted features from a URL string."""

    FEATURE_NAMES = [
        "having_ip_address", "url_length", "shortening_service",
        "having_at_symbol", "double_slash_redirecting", "prefix_suffix",
        "having_sub_domain", "ssl_state", "domain_reg_length",
        "favicon_external", "port_present", "https_token_in_domain",
        "request_url_ratio", "url_of_anchor_ratio", "links_in_tags",
        "sfh_suspicious", "submitting_to_email", "abnormal_url",
        "redirect_count", "on_mouseover", "right_click_disabled",
        "popup_window", "iframe_present", "age_of_domain",
        "dns_record", "web_traffic", "page_rank", "google_index",
        "links_pointing_to_page", "statistical_report",
    ]

    def extract(self, url: str) -> List[float]:
        url = url.strip()
        try:
            parsed = urllib.parse.urlparse(url if "://" in url else "http://" + url)
        except Exception:
            return [0.0] * 30

        domain = parsed.netloc.lower()
        path = parsed.path.lower()
        full = url.lower()

        features = [
            self._has_ip(domain),
            self._url_length(url),
            self._uses_shortener(domain),
            1.0 if "@" in url else 0.0,
            1.0 if "//" in path else 0.0,
            1.0 if "-" in domain else 0.0,
            self._sub_domain_count(domain),
            0.0 if full.startswith("https") else 1.0,   # 0=safe HTTPS
            self._domain_length_score(domain),
            0.0,  # favicon – not deterministic without fetching
            1.0 if parsed.port and parsed.port not in (80, 443) else 0.0,
            1.0 if "https" in domain else 0.0,
            0.0,  # request url ratio – needs page fetch
            0.0,  # anchor ratio – needs page fetch
            0.0,  # links in tags – needs page fetch
            0.0,  # SFH – needs page fetch
            1.0 if "mailto:" in full else 0.0,
            self._abnormal_url(domain, full),
            self._count_redirects(url),
            0.0, 0.0, 0.0, 0.0,  # JS-based features
            0.0,  # age of domain – needs WHOIS
            0.0,  # DNS record – needs DNS lookup
            0.0,  # web traffic – needs 3rd party
            0.0,  # page rank – needs 3rd party
            1.0 if full.startswith("http") else 0.0,
            0.0,  # links pointing – needs crawl
            self._suspicious_keywords(full),
        ]
        return features

    # ── private helpers ──────────────────────────────────────

    def _has_ip(self, domain: str) -> float:
        ipv4 = re.match(r"^\d{1,3}(\.\d{1,3}){3}", domain)
        return 1.0 if ipv4 else 0.0

    def _url_length(self, url: str) -> float:
        if len(url) < 54:
            return 0.0
        elif len(url) <= 75:
            return 0.5
        return 1.0

    def _uses_shortener(self, domain: str) -> float:
        return 1.0 if any(s in domain for s in SHORTENERS) else 0.0

    def _sub_domain_count(self, domain: str) -> float:
        parts = domain.replace("www.", "").split(".")
        return 0.0 if len(parts) <= 2 else (0.5 if len(parts) == 3 else 1.0)

    def _domain_length_score(self, domain: str) -> float:
        return 1.0 if len(domain) > 24 else 0.0

    def _abnormal_url(self, domain: str, full: str) -> float:
        return 1.0 if any(kw in full for kw in SUSPICIOUS_KEYWORDS) else 0.0

    def _count_redirects(self, url: str) -> float:
        count = url.count("//") - 1
        return min(count / 4.0, 1.0) if count > 0 else 0.0

    def _suspicious_keywords(self, full: str) -> float:
        hits = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in full)
        return min(hits / 3.0, 1.0)


# ─────────────────────────────────────────────────────────────
# Email Feature Extractor
# ─────────────────────────────────────────────────────────────

class EmailFeatureExtractor:
    """Extracts 30 features from email text."""

    FEATURE_NAMES = [
        "has_ip_link", "url_length_score", "link_shortener",
        "at_symbol_in_link", "double_slash_in_link", "suspicious_domain",
        "many_subdomains", "no_https", "urgent_language",
        "external_links", "mailto_present", "abnormal_link",
        "redirect_in_link", "caps_ratio", "exclamation_count",
        "question_count", "html_content", "spoofed_sender",
        "misspelled_brand", "generic_greeting", "request_credentials",
        "reward_language", "threat_language", "link_count",
        "image_only", "reply_to_mismatch", "no_unsubscribe",
        "suspicious_attachments", "body_length_score", "keyword_density",
    ]

    URGENT_WORDS = [
        "urgent", "immediately", "action required", "verify now",
        "suspended", "limited time", "expires", "warning",
        "act now", "last chance", "click here", "confirm your",
    ]
    REWARD_WORDS = [
        "congratulations", "winner", "won", "prize", "gift",
        "free", "claim", "reward", "bonus",
    ]
    THREAT_WORDS = [
        "suspended", "deleted", "terminated", "legal action",
        "law enforcement", "locked", "unauthorized",
    ]
    BRAND_MISSPELLS = {
        "paypal": ["paypall", "paypa1", "pay-pal"],
        "amazon": ["amaz0n", "amazon-support"],
        "google": ["g00gle", "googIe"],
        "microsoft": ["micros0ft", "microsft"],
        "apple": ["app1e", "apqle"],
    }

    def extract(self, text: str) -> List[float]:
        text_lower = text.lower()
        urls = re.findall(r"https?://\S+|www\.\S+", text_lower)
        url_ext = URLFeatureExtractor()
        url_features = url_ext.extract(urls[0]) if urls else [0.0] * 30

        words = text_lower.split()
        n_words = max(len(words), 1)
        n_chars = max(len(text), 1)

        caps_ratio = sum(1 for c in text if c.isupper()) / n_chars
        keyword_hits = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in text_lower)

        features = [
            url_features[0],   # ip in link
            url_features[1],   # url length
            url_features[2],   # shortener
            url_features[3],   # @ symbol
            url_features[4],   # double slash
            url_features[5],   # suspicious domain
            url_features[6],   # subdomains
            url_features[7],   # no https
            self._score(self.URGENT_WORDS, text_lower),
            min(len(urls) / 5.0, 1.0),
            1.0 if "mailto:" in text_lower else 0.0,
            url_features[17],  # abnormal url
            url_features[18],  # redirect
            min(caps_ratio * 2, 1.0),
            min(text.count("!") / 5.0, 1.0),
            min(text.count("?") / 5.0, 1.0),
            1.0 if re.search(r"<html|<body|<a href", text_lower) else 0.0,
            self._spoofed_sender(text_lower),
            self._misspelled_brand(text_lower),
            1.0 if re.search(r"dear (customer|user|member|sir|madam)", text_lower) else 0.0,
            1.0 if re.search(r"(enter|provide|confirm) (your )?(password|username|ssn|card)", text_lower) else 0.0,
            self._score(self.REWARD_WORDS, text_lower),
            self._score(self.THREAT_WORDS, text_lower),
            min(len(urls) / 3.0, 1.0),
            0.0,  # image-only – needs HTML parse
            0.0,  # reply-to mismatch – needs header
            1.0 if "unsubscribe" not in text_lower else 0.0,
            1.0 if re.search(r"\.(exe|zip|docm|xlsm|js|vbs)", text_lower) else 0.0,
            min(len(text) / 2000.0, 1.0),
            min(keyword_hits / 5.0, 1.0),
        ]
        return features

    def _score(self, word_list: List[str], text: str) -> float:
        hits = sum(1 for w in word_list if w in text)
        return min(hits / 3.0, 1.0)

    def _spoofed_sender(self, text: str) -> float:
        sender_match = re.search(r"from:.*?(\S+@\S+)", text)
        if not sender_match:
            return 0.0
        email_addr = sender_match.group(1)
        return 1.0 if any(kw in email_addr for kw in ["support", "noreply", "security"]) else 0.0

    def _misspelled_brand(self, text: str) -> float:
        for brand, variants in self.BRAND_MISSPELLS.items():
            if any(v in text for v in variants):
                return 1.0
        return 0.0


# ─────────────────────────────────────────────────────────────
# Message Feature Extractor
# ─────────────────────────────────────────────────────────────

class MessageFeatureExtractor:
    """Extracts 30 features from a free-text message/SMS."""

    FEATURE_NAMES = EmailFeatureExtractor.FEATURE_NAMES  # same schema

    def extract(self, text: str) -> List[float]:
        # Reuse email extractor — messages share most features
        return EmailFeatureExtractor().extract(text)


# ─────────────────────────────────────────────────────────────
# Unified Interface
# ─────────────────────────────────────────────────────────────

def extract_features(input_text: str, input_type: str) -> List[float]:
    """
    Route input to the correct extractor and return 30 features.

    Parameters
    ----------
    input_text : str
    input_type : str  – 'URL', 'Email', or 'Message'
    """
    if input_type == "URL":
        return URLFeatureExtractor().extract(input_text)
    elif input_type == "Email":
        return EmailFeatureExtractor().extract(input_text)
    else:
        return MessageFeatureExtractor().extract(input_text)


def get_feature_names(input_type: str) -> List[str]:
    if input_type == "URL":
        return URLFeatureExtractor.FEATURE_NAMES
    return EmailFeatureExtractor.FEATURE_NAMES
