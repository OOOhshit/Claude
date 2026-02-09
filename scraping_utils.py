#!/usr/bin/env python3
"""
Web Scraping Utilities - Solutions for common scraping challenges.

This module provides battle-tested solutions for:
  P1:  Playwright JS rendering with requests fallback
  P2:  Session/cookie management + cookie consent auto-accept
  P3:  User-Agent rotation + header/fingerprint randomization
  P4:  Adaptive retry with circuit breaker pattern
  P5:  CAPTCHA/Cloudflare/WAF detection
  P6:  Concurrent scraping with ThreadPoolExecutor
  P7:  URL deduplication and normalization
  P8:  Proxy support with rotation
  P9:  Content-type validation and encoding detection
  P10: Redirect tracking and loop detection
  P11: Incremental checkpoint/resume
  P12: Google Cache / Wayback Machine fallback
"""

import json
import hashlib
import logging
import os
import random
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urlunparse, parse_qs, urlencode, unquote

import requests
import urllib3
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# =============================================================================
# P3: User-Agent Rotation + Header Fingerprint Randomization
# =============================================================================

# Realistic User-Agent pool (Chrome, Firefox, Edge on Windows/Mac/Linux)
USER_AGENT_POOL = [
    # Chrome on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    # Chrome on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    # Firefox on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    # Firefox on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:122.0) Gecko/20100101 Firefox/122.0",
    # Edge on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    # Chrome on Linux
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
]

# Accept-Language variants for fingerprint diversity
ACCEPT_LANGUAGE_POOL = [
    "en-US,en;q=0.9",
    "en-US,en;q=0.9,fr;q=0.8",
    "en-GB,en;q=0.9,en-US;q=0.8",
    "en-US,en;q=0.9,de;q=0.8",
    "en-US,en;q=0.9,es;q=0.7",
    "en,en-US;q=0.9",
]


def get_randomized_headers(referer: Optional[str] = None) -> Dict[str, str]:
    """
    Generate a realistic, randomized set of browser headers.

    Each call produces slightly different fingerprints to avoid
    pattern-based bot detection. The UA, Accept-Language, and
    Sec-Fetch headers are varied while remaining internally consistent.
    """
    ua = random.choice(USER_AGENT_POOL)
    is_firefox = "Firefox" in ua
    is_chrome = "Chrome" in ua and "Edg" not in ua
    is_edge = "Edg" in ua

    headers = {
        "User-Agent": ua,
        "Accept-Language": random.choice(ACCEPT_LANGUAGE_POOL),
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    # Browser-specific Accept header variations
    if is_firefox:
        headers["Accept"] = "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
    else:
        headers["Accept"] = "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"

    # Sec-Fetch headers (Chrome/Edge only; Firefox uses them too since v90+)
    if not is_firefox or random.random() > 0.3:
        headers["Sec-Fetch-Dest"] = "document"
        headers["Sec-Fetch-Mode"] = "navigate"
        headers["Sec-Fetch-Site"] = "same-origin" if referer else "none"
        headers["Sec-Fetch-User"] = "?1"

    # Chrome-specific headers
    if is_chrome or is_edge:
        headers["Sec-Ch-Ua-Platform"] = random.choice(['"Windows"', '"macOS"', '"Linux"'])
        if is_edge:
            headers["Sec-Ch-Ua"] = '"Microsoft Edge";v="121", "Chromium";v="121", "Not A(Brand";v="99"'
        else:
            headers["Sec-Ch-Ua"] = '"Google Chrome";v="121", "Chromium";v="121", "Not A(Brand";v="99"'
        headers["Sec-Ch-Ua-Mobile"] = "?0"

    if referer:
        headers["Referer"] = referer

    # Randomize Cache-Control
    headers["Cache-Control"] = random.choice(["max-age=0", "no-cache", "max-age=0, no-cache"])

    return headers


# =============================================================================
# P4: Adaptive Retry with Circuit Breaker
# =============================================================================

class CircuitState(Enum):
    CLOSED = "closed"        # Normal operation -- requests go through
    OPEN = "open"            # Too many failures -- requests are blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class DomainHealth:
    """Tracks health metrics for a single domain."""
    domain: str
    consecutive_failures: int = 0
    total_requests: int = 0
    total_failures: int = 0
    total_successes: int = 0
    last_failure_time: float = 0.0
    circuit_state: CircuitState = CircuitState.CLOSED
    # Open circuit after this many consecutive failures
    failure_threshold: int = 5
    # How long (seconds) before trying again after circuit opens
    recovery_timeout: float = 60.0
    last_status_code: Optional[int] = None
    last_error: Optional[str] = None


class CircuitBreaker:
    """
    Per-domain circuit breaker to stop hammering dead/blocking sites.

    States:
      CLOSED    -> Normal. If consecutive failures reach threshold, go OPEN.
      OPEN      -> Block all requests. After recovery_timeout, go HALF_OPEN.
      HALF_OPEN -> Allow one test request. If success -> CLOSED, else -> OPEN.
    """

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self._domains: Dict[str, DomainHealth] = {}
        self._lock = threading.Lock()
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

    def _get_domain(self, url: str) -> str:
        return urlparse(url).netloc

    def _get_health(self, domain: str) -> DomainHealth:
        if domain not in self._domains:
            self._domains[domain] = DomainHealth(
                domain=domain,
                failure_threshold=self.failure_threshold,
                recovery_timeout=self.recovery_timeout,
            )
        return self._domains[domain]

    def can_request(self, url: str) -> bool:
        """Check if a request to this URL's domain is allowed."""
        domain = self._get_domain(url)
        with self._lock:
            health = self._get_health(domain)

            if health.circuit_state == CircuitState.CLOSED:
                return True
            elif health.circuit_state == CircuitState.OPEN:
                # Check if recovery timeout has elapsed
                elapsed = time.time() - health.last_failure_time
                if elapsed >= health.recovery_timeout:
                    health.circuit_state = CircuitState.HALF_OPEN
                    logger.info(f"[CIRCUIT] {domain}: OPEN -> HALF_OPEN (testing recovery)")
                    return True
                logger.debug(f"[CIRCUIT] {domain}: OPEN (retry in {health.recovery_timeout - elapsed:.0f}s)")
                return False
            elif health.circuit_state == CircuitState.HALF_OPEN:
                return True  # Allow the test request
            return False

    def record_success(self, url: str):
        """Record a successful request."""
        domain = self._get_domain(url)
        with self._lock:
            health = self._get_health(domain)
            health.consecutive_failures = 0
            health.total_successes += 1
            health.total_requests += 1
            if health.circuit_state == CircuitState.HALF_OPEN:
                health.circuit_state = CircuitState.CLOSED
                logger.info(f"[CIRCUIT] {domain}: HALF_OPEN -> CLOSED (recovered)")

    def record_failure(self, url: str, status_code: Optional[int] = None, error: Optional[str] = None):
        """Record a failed request."""
        domain = self._get_domain(url)
        with self._lock:
            health = self._get_health(domain)
            health.consecutive_failures += 1
            health.total_failures += 1
            health.total_requests += 1
            health.last_failure_time = time.time()
            health.last_status_code = status_code
            health.last_error = error

            if health.circuit_state == CircuitState.HALF_OPEN:
                health.circuit_state = CircuitState.OPEN
                logger.warning(f"[CIRCUIT] {domain}: HALF_OPEN -> OPEN (still failing)")
            elif health.consecutive_failures >= health.failure_threshold:
                health.circuit_state = CircuitState.OPEN
                logger.warning(
                    f"[CIRCUIT] {domain}: CLOSED -> OPEN "
                    f"({health.consecutive_failures} consecutive failures)"
                )

    def get_adaptive_delay(self, url: str) -> float:
        """Calculate adaptive delay based on domain health."""
        domain = self._get_domain(url)
        with self._lock:
            health = self._get_health(domain)
            base_delay = 1.0
            # Increase delay as failures accumulate
            if health.consecutive_failures > 0:
                base_delay = min(1.0 * (2 ** health.consecutive_failures), 30.0)
            # Add jitter
            return base_delay + random.uniform(0.5, 2.0)

    def get_stats(self) -> Dict[str, dict]:
        """Get health statistics for all domains."""
        with self._lock:
            return {
                domain: {
                    "state": health.circuit_state.value,
                    "total_requests": health.total_requests,
                    "successes": health.total_successes,
                    "failures": health.total_failures,
                    "consecutive_failures": health.consecutive_failures,
                    "success_rate": (
                        health.total_successes / health.total_requests
                        if health.total_requests > 0
                        else 0.0
                    ),
                }
                for domain, health in self._domains.items()
            }


def request_with_adaptive_retry(
    session: requests.Session,
    url: str,
    circuit_breaker: CircuitBreaker,
    max_retries: int = 4,
    verify_ssl: bool = True,
    proxy: Optional[str] = None,
) -> Optional[requests.Response]:
    """
    Make an HTTP request with adaptive retry, circuit breaker, and backoff.

    Differentiates between error types:
      - 403:  Likely bot-detected. Try with different headers, then give up.
      - 429:  Rate limited. Back off aggressively.
      - 5xx:  Server error. Retry with exponential backoff.
      - Timeout/Connection: Retry with backoff.
      - SSL:  Don't retry (fundamental config issue).
    """
    if not circuit_breaker.can_request(url):
        logger.warning(f"[CIRCUIT] Request blocked for {url} (circuit open)")
        return None

    proxies = {"http": proxy, "https": proxy} if proxy else None

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                wait_time = circuit_breaker.get_adaptive_delay(url)
                logger.info(f"[RETRY] Attempt {attempt + 1}/{max_retries} for {url} (waiting {wait_time:.1f}s)")
                time.sleep(wait_time)

            # Rotate headers on each retry
            session.headers.update(get_randomized_headers())

            response = session.get(
                url,
                timeout=25,
                verify=verify_ssl,
                allow_redirects=True,
                proxies=proxies,
            )

            if response.status_code == 200:
                circuit_breaker.record_success(url)
                return response
            elif response.status_code == 403:
                logger.warning(f"[BLOCKED] 403 Forbidden: {url}")
                # Try once more with completely fresh headers
                if attempt == 0:
                    session.headers.update(get_randomized_headers())
                    continue
                circuit_breaker.record_failure(url, 403, "Forbidden")
                return None
            elif response.status_code == 429:
                logger.warning(f"[RATE LIMITED] 429 for {url}")
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        time.sleep(min(float(retry_after), 60.0))
                    except ValueError:
                        time.sleep(10)
                circuit_breaker.record_failure(url, 429, "Rate limited")
                continue
            elif response.status_code >= 500:
                logger.warning(f"[SERVER ERROR] {response.status_code} for {url}")
                circuit_breaker.record_failure(url, response.status_code, "Server error")
                continue
            else:
                # Other status codes (301, 302 handled by allow_redirects)
                circuit_breaker.record_success(url)
                return response

        except requests.exceptions.SSLError as e:
            logger.error(f"[SSL ERROR] {url}: {e}")
            circuit_breaker.record_failure(url, None, f"SSL: {e}")
            return None  # Don't retry SSL errors

        except requests.exceptions.Timeout:
            logger.warning(f"[TIMEOUT] {url} (attempt {attempt + 1})")
            circuit_breaker.record_failure(url, None, "Timeout")
            continue

        except requests.exceptions.ConnectionError as e:
            logger.warning(f"[CONNECTION] {url}: {e}")
            circuit_breaker.record_failure(url, None, f"Connection: {e}")
            continue

        except Exception as e:
            logger.error(f"[ERROR] Unexpected for {url}: {e}")
            circuit_breaker.record_failure(url, None, str(e))
            if attempt == max_retries - 1:
                return None

    return None


# =============================================================================
# P5: CAPTCHA / Cloudflare / WAF Detection
# =============================================================================

WAF_SIGNATURES = {
    "cloudflare": [
        "cf-browser-verification",
        "cloudflare",
        "cf-challenge",
        "ray id",
        "checking your browser",
        "enable javascript and cookies to continue",
        "__cf_bm",
        "cf_clearance",
        "just a moment",
    ],
    "akamai": [
        "akamai",
        "ak_bmsc",
        "bm_sv",
        "reference #",
        "access denied",
    ],
    "incapsula": [
        "incapsula",
        "imperva",
        "_incap_",
        "visid_incap",
    ],
    "captcha": [
        "recaptcha",
        "hcaptcha",
        "g-recaptcha",
        "h-captcha",
        "captcha",
        "solve the challenge",
    ],
    "generic_block": [
        "access denied",
        "403 forbidden",
        "request blocked",
        "bot detected",
        "automated access",
        "please verify you are human",
        "suspicious activity",
    ],
}


@dataclass
class WAFDetectionResult:
    """Result of WAF/CAPTCHA detection analysis."""
    is_blocked: bool = False
    waf_type: Optional[str] = None
    has_captcha: bool = False
    confidence: float = 0.0
    signals: List[str] = field(default_factory=list)
    recommendation: str = ""


def detect_waf_or_captcha(response: requests.Response) -> WAFDetectionResult:
    """
    Analyze an HTTP response to detect WAF blocks, CAPTCHAs, or bot detection.

    Checks:
      1. Response status code (403, 503 with challenge)
      2. Response headers (Cloudflare-specific headers, Akamai cookies)
      3. Response body signatures (challenge scripts, CAPTCHA forms)
      4. Content length (suspiciously short = likely block page)
    """
    result = WAFDetectionResult()
    body_lower = response.text.lower() if response.text else ""
    headers_str = str(response.headers).lower()

    # Check status codes commonly used by WAFs
    if response.status_code in (403, 503):
        result.signals.append(f"Suspicious status code: {response.status_code}")
        result.confidence += 0.3

    # Check for Cloudflare headers
    if "cf-ray" in response.headers or "cf-cache-status" in response.headers:
        if response.status_code in (403, 503):
            result.signals.append("Cloudflare headers + block status")
            result.waf_type = "cloudflare"
            result.confidence += 0.4

    # Check for very short response that indicates a block page
    if len(body_lower) < 5000 and response.status_code in (403, 503):
        result.signals.append("Suspiciously short response body")
        result.confidence += 0.2

    # Scan body for WAF signatures
    for waf_name, signatures in WAF_SIGNATURES.items():
        matches = [sig for sig in signatures if sig in body_lower or sig in headers_str]
        if matches:
            if waf_name == "captcha":
                result.has_captcha = True
                result.signals.append(f"CAPTCHA detected: {', '.join(matches[:3])}")
                result.confidence += 0.5
            else:
                if not result.waf_type:
                    result.waf_type = waf_name
                result.signals.append(f"{waf_name} signatures: {', '.join(matches[:3])}")
                result.confidence += 0.3

    result.confidence = min(result.confidence, 1.0)
    result.is_blocked = result.confidence >= 0.5

    # Set recommendation
    if result.is_blocked:
        if result.has_captcha:
            result.recommendation = "Use Playwright with stealth mode to solve interactively"
        elif result.waf_type == "cloudflare":
            result.recommendation = "Use Playwright to pass Cloudflare challenge"
        else:
            result.recommendation = "Try Playwright browser rendering or proxy rotation"
    else:
        result.recommendation = "No WAF detected, standard scraping should work"

    return result


# =============================================================================
# P7: URL Deduplication and Normalization
# =============================================================================

class URLNormalizer:
    """
    Normalizes and deduplicates URLs to avoid scraping the same content twice.

    Handles:
      - Trailing slashes
      - Fragment removal (#section)
      - Query parameter sorting
      - Protocol normalization
      - www prefix normalization
      - Case normalization for scheme/host
    """

    def __init__(self):
        self._seen_urls: Set[str] = set()
        self._seen_content_hashes: Set[str] = set()
        self._lock = threading.Lock()

    def normalize(self, url: str) -> str:
        """Normalize a URL to its canonical form."""
        parsed = urlparse(url)

        # Lowercase scheme and host
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()

        # Remove default ports
        if netloc.endswith(":80") and scheme == "http":
            netloc = netloc[:-3]
        elif netloc.endswith(":443") and scheme == "https":
            netloc = netloc[:-4]

        # Normalize path: remove trailing slash (except root), decode then re-encode
        path = parsed.path
        if path != "/" and path.endswith("/"):
            path = path.rstrip("/")
        if not path:
            path = "/"

        # Sort query parameters for consistency
        query = parsed.query
        if query:
            params = parse_qs(query, keep_blank_values=True)
            sorted_params = sorted(params.items())
            query = urlencode(sorted_params, doseq=True)

        # Remove fragment
        normalized = urlunparse((scheme, netloc, path, parsed.params, query, ""))
        return normalized

    def is_duplicate(self, url: str) -> bool:
        """Check if this URL (after normalization) has been seen before."""
        normalized = self.normalize(url)
        with self._lock:
            if normalized in self._seen_urls:
                return True
            self._seen_urls.add(normalized)
            return False

    def is_content_duplicate(self, content: str) -> bool:
        """Check if content is a duplicate based on hash."""
        content_hash = hashlib.md5(content.encode("utf-8", errors="ignore")).hexdigest()
        with self._lock:
            if content_hash in self._seen_content_hashes:
                return True
            self._seen_content_hashes.add(content_hash)
            return False

    def reset(self):
        """Clear all seen URLs and content hashes."""
        with self._lock:
            self._seen_urls.clear()
            self._seen_content_hashes.clear()


# =============================================================================
# P8: Proxy Support with Rotation
# =============================================================================

class ProxyRotator:
    """
    Manages a pool of proxies and rotates them per-request or per-domain.

    Supports:
      - Loading proxies from a file (one per line: http://host:port)
      - Health checking and blacklisting bad proxies
      - Round-robin or random rotation
      - Fallback to direct connection when all proxies fail
    """

    def __init__(self, proxy_file: Optional[str] = None, proxies: Optional[List[str]] = None):
        self._proxies: List[str] = []
        self._blacklisted: Set[str] = set()
        self._index = 0
        self._lock = threading.Lock()

        if proxy_file and os.path.exists(proxy_file):
            with open(proxy_file, "r") as f:
                self._proxies = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            logger.info(f"[PROXY] Loaded {len(self._proxies)} proxies from {proxy_file}")
        elif proxies:
            self._proxies = proxies
            logger.info(f"[PROXY] Initialized with {len(self._proxies)} proxies")

    @property
    def available(self) -> bool:
        """True if there are usable proxies."""
        return len(self._proxies) > len(self._blacklisted)

    def get_next(self) -> Optional[str]:
        """Get the next available proxy (round-robin)."""
        if not self._proxies:
            return None
        with self._lock:
            attempts = 0
            while attempts < len(self._proxies):
                proxy = self._proxies[self._index % len(self._proxies)]
                self._index += 1
                attempts += 1
                if proxy not in self._blacklisted:
                    return proxy
        logger.warning("[PROXY] All proxies blacklisted, falling back to direct")
        return None

    def get_random(self) -> Optional[str]:
        """Get a random available proxy."""
        if not self._proxies:
            return None
        available = [p for p in self._proxies if p not in self._blacklisted]
        if not available:
            logger.warning("[PROXY] All proxies blacklisted, falling back to direct")
            return None
        return random.choice(available)

    def blacklist(self, proxy: str):
        """Blacklist a non-working proxy."""
        with self._lock:
            self._blacklisted.add(proxy)
            remaining = len(self._proxies) - len(self._blacklisted)
            logger.warning(f"[PROXY] Blacklisted {proxy} ({remaining} remaining)")

    def reset(self):
        """Clear all blacklists."""
        with self._lock:
            self._blacklisted.clear()


# =============================================================================
# P9: Content-Type Validation and Encoding Detection
# =============================================================================

ALLOWED_CONTENT_TYPES = {
    "text/html",
    "application/xhtml+xml",
    "text/plain",
    "application/xml",
    "text/xml",
}

MAX_RESPONSE_SIZE = 10 * 1024 * 1024  # 10 MB


def validate_response(response: requests.Response) -> Tuple[bool, str]:
    """
    Validate that a response is suitable for HTML parsing.

    Returns:
        (is_valid, reason)
    """
    # Check content type
    content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        if content_type == "application/pdf":
            return False, f"PDF document (use PDF extractor instead)"
        return False, f"Unsupported content type: {content_type}"

    # Check content length
    content_length = response.headers.get("Content-Length")
    if content_length:
        try:
            size = int(content_length)
            if size > MAX_RESPONSE_SIZE:
                return False, f"Response too large: {size / 1024 / 1024:.1f} MB"
        except ValueError:
            pass

    # Check actual content size
    if len(response.content) > MAX_RESPONSE_SIZE:
        return False, f"Response body too large: {len(response.content) / 1024 / 1024:.1f} MB"

    return True, "OK"


def detect_encoding(response: requests.Response) -> str:
    """
    Detect the correct character encoding for a response.

    Priority: HTTP header > BOM > meta tag > chardet > utf-8 fallback.
    """
    # 1. Check Content-Type header
    content_type = response.headers.get("Content-Type", "")
    match = re.search(r"charset=([^\s;]+)", content_type, re.IGNORECASE)
    if match:
        return match.group(1).strip('"\'')

    # 2. Check HTML meta tags
    raw_content = response.content[:4096]
    meta_match = re.search(
        rb'<meta[^>]+charset=["\']?([^"\'\s;>]+)', raw_content, re.IGNORECASE
    )
    if meta_match:
        return meta_match.group(1).decode("ascii", errors="ignore")

    # 3. Check BOM
    if raw_content.startswith(b"\xef\xbb\xbf"):
        return "utf-8"
    if raw_content.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"

    # 4. Use response.apparent_encoding (uses chardet internally)
    if response.apparent_encoding:
        return response.apparent_encoding

    # 5. Default
    return "utf-8"


# =============================================================================
# P10: Redirect Tracking and Loop Detection
# =============================================================================

MAX_REDIRECTS = 10


def check_redirects(response: requests.Response) -> Tuple[bool, List[str]]:
    """
    Analyze redirect chain for issues (loops, excessive redirects).

    Returns:
        (is_ok, redirect_chain_urls)
    """
    chain = [r.url for r in response.history]
    chain.append(response.url)

    if len(response.history) > MAX_REDIRECTS:
        logger.warning(f"[REDIRECT] Excessive redirects ({len(response.history)}): {chain[0]} -> {chain[-1]}")
        return False, chain

    # Detect loops
    seen = set()
    for url in chain:
        normalized = urlparse(url)._replace(fragment="").geturl()
        if normalized in seen:
            logger.warning(f"[REDIRECT] Loop detected in chain: {chain}")
            return False, chain
        seen.add(normalized)

    if len(chain) > 1:
        logger.debug(f"[REDIRECT] {chain[0]} -> {chain[-1]} ({len(chain) - 1} redirects)")

    return True, chain


# =============================================================================
# P11: Incremental Checkpoint / Resume
# =============================================================================

CHECKPOINT_FILE = "scraping_checkpoint.json"


@dataclass
class ScrapeCheckpoint:
    """Saves scraping progress for resume capability."""
    completed_companies: List[str] = field(default_factory=list)
    completed_urls: List[str] = field(default_factory=list)
    failed_urls: List[str] = field(default_factory=list)
    scraped_data: List[dict] = field(default_factory=list)
    timestamp: str = ""
    total_companies: int = 0

    def save(self, filepath: str = CHECKPOINT_FILE):
        self.timestamp = datetime.now().isoformat()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
        logger.info(
            f"[CHECKPOINT] Saved: {len(self.completed_companies)}/{self.total_companies} companies, "
            f"{len(self.completed_urls)} URLs scraped"
        )

    @classmethod
    def load(cls, filepath: str = CHECKPOINT_FILE) -> Optional["ScrapeCheckpoint"]:
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            checkpoint = cls(**data)
            logger.info(
                f"[CHECKPOINT] Resumed from {checkpoint.timestamp}: "
                f"{len(checkpoint.completed_companies)} companies done"
            )
            return checkpoint
        except Exception as e:
            logger.warning(f"[CHECKPOINT] Could not load checkpoint: {e}")
            return None

    def is_company_done(self, company_name: str) -> bool:
        return company_name in self.completed_companies

    def is_url_done(self, url: str) -> bool:
        return url in self.completed_urls

    def mark_company_done(self, company_name: str):
        if company_name not in self.completed_companies:
            self.completed_companies.append(company_name)

    def mark_url_done(self, url: str):
        if url not in self.completed_urls:
            self.completed_urls.append(url)

    def add_scraped_data(self, data: dict):
        self.scraped_data.append(data)

    def clear(self, filepath: str = CHECKPOINT_FILE):
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info("[CHECKPOINT] Cleared checkpoint file")


# =============================================================================
# P12: Google Cache / Wayback Machine Fallback
# =============================================================================

def fetch_from_wayback(url: str, session: Optional[requests.Session] = None) -> Optional[str]:
    """
    Fetch a page from the Wayback Machine as a fallback when direct scraping fails.

    Uses the Wayback Machine Availability API to find the most recent snapshot.
    """
    s = session or requests.Session()
    api_url = f"https://archive.org/wayback/available?url={url}"

    try:
        resp = s.get(api_url, timeout=15)
        if resp.status_code != 200:
            return None

        data = resp.json()
        snapshot = data.get("archived_snapshots", {}).get("closest")
        if not snapshot or not snapshot.get("available"):
            logger.info(f"[WAYBACK] No snapshot available for {url}")
            return None

        archive_url = snapshot["url"]
        timestamp = snapshot.get("timestamp", "unknown")
        logger.info(f"[WAYBACK] Found snapshot from {timestamp}: {archive_url}")

        page_resp = s.get(archive_url, timeout=20)
        if page_resp.status_code == 200:
            # Remove Wayback Machine toolbar/injection
            content = page_resp.text
            content = re.sub(
                r'<!-- BEGIN WAYBACK TOOLBAR INSERT -->.*?<!-- END WAYBACK TOOLBAR INSERT -->',
                '', content, flags=re.DOTALL
            )
            return content

    except Exception as e:
        logger.warning(f"[WAYBACK] Error fetching {url}: {e}")

    return None


def fetch_from_google_cache(url: str, session: Optional[requests.Session] = None) -> Optional[str]:
    """
    Attempt to fetch a page from Google's cache as a fallback.

    Note: Google Cache has been largely deprecated. This is a best-effort attempt.
    """
    s = session or requests.Session()
    cache_url = f"https://webcache.googleusercontent.com/search?q=cache:{url}"

    try:
        s.headers.update(get_randomized_headers())
        resp = s.get(cache_url, timeout=15)
        if resp.status_code == 200 and len(resp.text) > 1000:
            logger.info(f"[GOOGLE CACHE] Found cached version for {url}")
            return resp.text
    except Exception as e:
        logger.debug(f"[GOOGLE CACHE] Not available for {url}: {e}")

    return None


def fetch_with_fallback(url: str, session: Optional[requests.Session] = None) -> Optional[str]:
    """Try Google Cache first (faster), then Wayback Machine."""
    content = fetch_from_google_cache(url, session)
    if content:
        return content
    return fetch_from_wayback(url, session)


# =============================================================================
# P1: Playwright-based JavaScript Rendering
# =============================================================================

class PlaywrightFetcher:
    """
    Fetches pages using a real headless browser via Playwright.

    This handles:
      - JavaScript-rendered content (React, Angular, Vue sites)
      - Cloudflare/Akamai challenges (browser fingerprint passes)
      - Cookie consent popups (auto-dismisses common patterns)
      - Dynamic content loading (waits for network idle)

    Falls back gracefully to requests+BeautifulSoup if Playwright is not installed.
    """

    def __init__(self, headless: bool = True):
        self.headless = headless
        self._browser = None
        self._context = None
        self._playwright = None
        self._available = False
        self._checked = False

    def is_available(self) -> bool:
        """Check if Playwright is installed and browsers are available."""
        if self._checked:
            return self._available
        self._checked = True
        try:
            from playwright.sync_api import sync_playwright
            self._available = True
            logger.info("[PLAYWRIGHT] Playwright is available")
        except ImportError:
            self._available = False
            logger.warning(
                "[PLAYWRIGHT] Playwright not installed. "
                "Install with: pip install playwright && playwright install chromium"
            )
        return self._available

    def _ensure_browser(self):
        """Lazily start the browser only when first needed."""
        if self._browser is not None:
            return

        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
        )
        # Create a persistent context with realistic viewport and locale
        self._context = self._browser.new_context(
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
            timezone_id="America/New_York",
            user_agent=random.choice(USER_AGENT_POOL),
        )
        # Stealth: remove webdriver flag
        self._context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
            Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
            window.chrome = { runtime: {} };
        """)
        logger.info("[PLAYWRIGHT] Browser launched with stealth settings")

    def _dismiss_cookie_consent(self, page) -> bool:
        """
        P2: Auto-dismiss cookie consent popups.

        Tries common selectors for Accept/Agree/OK buttons.
        """
        consent_selectors = [
            # Common consent button patterns
            'button:has-text("Accept All")',
            'button:has-text("Accept all")',
            'button:has-text("Accept Cookies")',
            'button:has-text("Accept cookies")',
            'button:has-text("I Accept")',
            'button:has-text("I agree")',
            'button:has-text("Agree")',
            'button:has-text("Allow All")',
            'button:has-text("Allow all")',
            'button:has-text("OK")',
            'button:has-text("Got it")',
            'button:has-text("Consent")',
            # Common ID/class patterns
            '#onetrust-accept-btn-handler',
            '#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll',
            '.cookie-accept',
            '.cookie-consent-accept',
            '.accept-cookies',
            '[data-testid="cookie-accept"]',
            '[aria-label="Accept cookies"]',
            '#accept-cookies',
            '.cc-accept',
            '.cc-btn.cc-allow',
        ]
        for selector in consent_selectors:
            try:
                btn = page.locator(selector).first
                if btn.is_visible(timeout=500):
                    btn.click()
                    logger.info(f"[COOKIE CONSENT] Dismissed with selector: {selector}")
                    return True
            except Exception:
                continue
        return False

    def fetch(self, url: str, wait_for_selector: Optional[str] = None,
              wait_seconds: float = 3.0) -> Optional[str]:
        """
        Fetch a page with full JavaScript rendering.

        Args:
            url: The URL to fetch
            wait_for_selector: Optional CSS selector to wait for
            wait_seconds: Seconds to wait for dynamic content after load

        Returns:
            Rendered HTML string, or None on failure
        """
        if not self.is_available():
            return None

        try:
            self._ensure_browser()
            page = self._context.new_page()

            try:
                # Navigate with timeout
                page.goto(url, wait_until="domcontentloaded", timeout=30000)

                # Wait for network to settle
                try:
                    page.wait_for_load_state("networkidle", timeout=10000)
                except Exception:
                    pass  # Some sites never reach networkidle

                # Dismiss cookie consent if present
                self._dismiss_cookie_consent(page)

                # Wait for specific selector if provided
                if wait_for_selector:
                    try:
                        page.wait_for_selector(wait_for_selector, timeout=5000)
                    except Exception:
                        logger.debug(f"[PLAYWRIGHT] Selector '{wait_for_selector}' not found")

                # Small delay for any final JS rendering
                page.wait_for_timeout(int(wait_seconds * 1000))

                # Scroll down to trigger lazy-loaded content
                page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
                page.wait_for_timeout(1000)
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(1000)

                html = page.content()
                logger.info(f"[PLAYWRIGHT] Successfully rendered {url} ({len(html)} chars)")
                return html

            finally:
                page.close()

        except Exception as e:
            logger.error(f"[PLAYWRIGHT] Error fetching {url}: {e}")
            return None

    def close(self):
        """Clean up browser resources."""
        try:
            if self._context:
                self._context.close()
            if self._browser:
                self._browser.close()
            if self._playwright:
                self._playwright.stop()
            logger.info("[PLAYWRIGHT] Browser closed")
        except Exception as e:
            logger.debug(f"[PLAYWRIGHT] Error closing: {e}")


# =============================================================================
# P2: Session / Cookie Management
# =============================================================================

class SessionManager:
    """
    Manages HTTP sessions with proper cookie handling and persistence.

    Features:
      - Cookie jar persistence across requests
      - Per-domain session isolation
      - Cookie consent pre-acceptance
      - Session rotation to avoid tracking
    """

    def __init__(self, verify_ssl: bool = True):
        self.verify_ssl = verify_ssl
        self._sessions: Dict[str, requests.Session] = {}
        self._lock = threading.Lock()

        if not verify_ssl:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def get_session(self, url: str) -> requests.Session:
        """Get or create a session for this domain. Sessions are reused per-domain."""
        domain = urlparse(url).netloc
        with self._lock:
            if domain not in self._sessions:
                session = requests.Session()
                session.headers.update(get_randomized_headers())
                session.verify = self.verify_ssl
                # Set reasonable defaults
                session.max_redirects = MAX_REDIRECTS
                self._sessions[domain] = session
                logger.debug(f"[SESSION] Created new session for {domain}")
            return self._sessions[domain]

    def rotate_session(self, url: str) -> requests.Session:
        """Force a new session for this domain (useful after blocks)."""
        domain = urlparse(url).netloc
        with self._lock:
            session = requests.Session()
            session.headers.update(get_randomized_headers())
            session.verify = self.verify_ssl
            session.max_redirects = MAX_REDIRECTS
            self._sessions[domain] = session
            logger.info(f"[SESSION] Rotated session for {domain}")
            return session

    def close_all(self):
        """Close all sessions."""
        with self._lock:
            for session in self._sessions.values():
                session.close()
            self._sessions.clear()


# =============================================================================
# P6: Concurrent Scraping with ThreadPoolExecutor
# =============================================================================

class ConcurrentScraper:
    """
    Orchestrates concurrent scraping across multiple companies/URLs.

    Uses ThreadPoolExecutor for parallel scraping while respecting
    per-domain rate limits via the circuit breaker and delay mechanisms.
    """

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self._domain_locks: Dict[str, threading.Lock] = {}
        self._domain_last_request: Dict[str, float] = {}
        self._global_lock = threading.Lock()

    def _get_domain_lock(self, domain: str) -> threading.Lock:
        with self._global_lock:
            if domain not in self._domain_locks:
                self._domain_locks[domain] = threading.Lock()
            return self._domain_locks[domain]

    def enforce_rate_limit(self, url: str, delay: float = 1.5):
        """Ensure minimum delay between requests to the same domain."""
        domain = urlparse(url).netloc
        lock = self._get_domain_lock(domain)
        with lock:
            now = time.time()
            last = self._domain_last_request.get(domain, 0)
            wait = delay - (now - last)
            if wait > 0:
                time.sleep(wait)
            self._domain_last_request[domain] = time.time()

    def scrape_companies_parallel(self, companies: List[dict], scrape_func) -> List[dict]:
        """
        Scrape multiple companies in parallel.

        Args:
            companies: List of company dicts with 'name' and 'url' keys
            scrape_func: Callable(company_dict) -> list of scraped data dicts

        Returns:
            Combined list of all scraped data
        """
        all_results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_company = {
                executor.submit(scrape_func, company): company
                for company in companies
            }
            for future in as_completed(future_to_company):
                company = future_to_company[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    logger.info(
                        f"[CONCURRENT] Completed {company['name']}: "
                        f"{len(results)} pages scraped"
                    )
                except Exception as e:
                    logger.error(f"[CONCURRENT] Failed {company['name']}: {e}")
        return all_results


# =============================================================================
# Summary: Putting It All Together
# =============================================================================

class EnhancedScrapingEngine:
    """
    Unified scraping engine combining all 12 challenge solutions.

    Usage:
        engine = EnhancedScrapingEngine(verify_ssl=True)
        html = engine.smart_fetch("https://example.com")
        soup = BeautifulSoup(html, 'html.parser')
        engine.close()

    The smart_fetch method automatically:
      1. Checks circuit breaker (P4)
      2. Checks URL deduplication (P7)
      3. Tries fast requests+headers first (P3, P9, P10)
      4. Detects WAF/CAPTCHA blocks (P5)
      5. Falls back to Playwright if blocked (P1, P2)
      6. Falls back to Wayback Machine as last resort (P12)
      7. Saves checkpoint periodically (P11)
    """

    def __init__(
        self,
        verify_ssl: bool = True,
        use_playwright: bool = True,
        proxy_file: Optional[str] = None,
        max_workers: int = 4,
        checkpoint_enabled: bool = True,
    ):
        self.session_manager = SessionManager(verify_ssl=verify_ssl)
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        self.url_normalizer = URLNormalizer()
        self.proxy_rotator = ProxyRotator(proxy_file=proxy_file)
        self.concurrent = ConcurrentScraper(max_workers=max_workers)
        self.checkpoint = ScrapeCheckpoint() if checkpoint_enabled else None
        self.verify_ssl = verify_ssl

        self._playwright_fetcher: Optional[PlaywrightFetcher] = None
        if use_playwright:
            self._playwright_fetcher = PlaywrightFetcher(headless=True)

        # Statistics
        self.stats = {
            "total_requests": 0,
            "requests_success": 0,
            "requests_failed": 0,
            "playwright_used": 0,
            "wayback_used": 0,
            "waf_detected": 0,
            "duplicates_skipped": 0,
        }

    def smart_fetch(self, url: str, company_name: str = "") -> Optional[str]:
        """
        Intelligently fetch a URL using the best available method.

        Strategy:
          1. requests (fast) -> check for WAF
          2. If WAF detected -> Playwright (renders JS, passes challenges)
          3. If all fail -> Wayback Machine (archived content)
        """
        self.stats["total_requests"] += 1

        # P7: Check URL dedup
        if self.url_normalizer.is_duplicate(url):
            self.stats["duplicates_skipped"] += 1
            logger.debug(f"[DEDUP] Skipping duplicate URL: {url}")
            return None

        # P4: Check circuit breaker
        if not self.circuit_breaker.can_request(url):
            logger.warning(f"[CIRCUIT] Skipping {url} (circuit open for domain)")
            return None

        # P11: Check checkpoint
        if self.checkpoint and self.checkpoint.is_url_done(url):
            logger.debug(f"[CHECKPOINT] Skipping already-scraped URL: {url}")
            return None

        # P8: Get proxy if available
        proxy = self.proxy_rotator.get_random() if self.proxy_rotator.available else None

        # --- Strategy 1: Fast requests-based fetch ---
        session = self.session_manager.get_session(url)
        response = request_with_adaptive_retry(
            session, url, self.circuit_breaker,
            max_retries=3, verify_ssl=self.verify_ssl, proxy=proxy,
        )

        if response is not None:
            # P9: Validate content type
            is_valid, reason = validate_response(response)
            if not is_valid:
                logger.warning(f"[VALIDATION] {url}: {reason}")
                return None

            # P10: Check redirects
            redirect_ok, chain = check_redirects(response)
            if not redirect_ok:
                return None

            # P5: Check for WAF / CAPTCHA
            waf_result = detect_waf_or_captcha(response)
            if not waf_result.is_blocked:
                # Success via requests
                self.stats["requests_success"] += 1
                # Detect encoding
                encoding = detect_encoding(response)
                response.encoding = encoding
                html = response.text
                if self.checkpoint:
                    self.checkpoint.mark_url_done(url)
                return html
            else:
                logger.warning(
                    f"[WAF] Detected {waf_result.waf_type or 'unknown'} on {url}: "
                    f"{', '.join(waf_result.signals[:3])}"
                )
                self.stats["waf_detected"] += 1
                # Fall through to Playwright

        # --- Strategy 2: Playwright (handles JS + WAF) ---
        if self._playwright_fetcher and self._playwright_fetcher.is_available():
            logger.info(f"[FALLBACK] Trying Playwright for {url}")
            html = self._playwright_fetcher.fetch(url)
            if html and len(html) > 500:
                self.stats["playwright_used"] += 1
                self.circuit_breaker.record_success(url)
                if self.checkpoint:
                    self.checkpoint.mark_url_done(url)
                return html

        # --- Strategy 3: Wayback Machine fallback ---
        logger.info(f"[FALLBACK] Trying Wayback Machine for {url}")
        html = fetch_with_fallback(url, session)
        if html:
            self.stats["wayback_used"] += 1
            if self.checkpoint:
                self.checkpoint.mark_url_done(url)
            return html

        self.stats["requests_failed"] += 1
        logger.error(f"[FAILED] All strategies exhausted for {url}")
        return None

    def save_checkpoint(self):
        """Save current progress."""
        if self.checkpoint:
            self.checkpoint.save()

    def load_checkpoint(self) -> bool:
        """Load previous checkpoint. Returns True if loaded."""
        loaded = ScrapeCheckpoint.load()
        if loaded:
            self.checkpoint = loaded
            return True
        return False

    def get_stats(self) -> dict:
        """Get scraping statistics."""
        return {
            **self.stats,
            "circuit_breaker": self.circuit_breaker.get_stats(),
        }

    def close(self):
        """Clean up all resources."""
        if self._playwright_fetcher:
            self._playwright_fetcher.close()
        self.session_manager.close_all()
        logger.info(f"[ENGINE] Closed. Stats: {json.dumps(self.stats, indent=2)}")
