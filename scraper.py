#!/usr/bin/env python3
"""
Oil Companies Market Scanner

Market-focused scraper using a 3-layer strategy:
  Layer 1: requests + randomized headers (fast, low resource)
  Layer 2: Playwright headless browser (JS rendering, WAF bypass)
  Layer 3: Wayback Machine / Google Cache (last resort fallback)

Focuses on oil & energy market segments only (no technology/AI/IT search).
All 12 scraping challenges are handled via scraping_utils.py.
"""

import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, unquote
from urllib.robotparser import RobotFileParser
import time
import logging
from typing import List, Dict, Set, Optional
import re
from dataclasses import dataclass, asdict
import random
import urllib3

from scraping_utils import (
    EnhancedScrapingEngine,
    get_randomized_headers,
    detect_waf_or_captcha,
    ScrapeCheckpoint,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ScrapedContent:
    company: str
    url: str
    title: str
    content: str
    page_type: str  # 'technology', 'innovation', 'market', etc.

@dataclass
class CompanyAnalysis:
    company: str
    market_presence: str
    technologies: str
    innovations: str
    summary: str

class OilCompanyScanner:
    def __init__(self, verify_ssl: bool = True, use_playwright: bool = True,
                 proxy_file: Optional[str] = None, max_workers: int = 3,
                 resume: bool = True):
        self.verify_ssl = verify_ssl

        # Initialize the enhanced scraping engine (all 12 solutions)
        self.engine = EnhancedScrapingEngine(
            verify_ssl=verify_ssl,
            use_playwright=use_playwright,
            proxy_file=proxy_file,
            max_workers=max_workers,
            checkpoint_enabled=resume,
        )

        # Keep a lightweight requests session for robots.txt checks
        self._robots_session = requests.Session()
        self._robots_session.headers.update(get_randomized_headers())
        self._robots_session.verify = verify_ssl
        if not verify_ssl:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            logger.warning("[SSL] SSL certificate verification is DISABLED")

        self.scraped_content: List[ScrapedContent] = []
        self.analysis_results: List[CompanyAnalysis] = []
        self.market_keywords = self.load_market_keywords()
        self.visited_urls: List[Dict] = []
        self.robots_cache: Dict[str, RobotFileParser] = {}
        self.blocked_sites: List[Dict] = []
        self._resume = resume

    def load_companies(self, filename: str = 'companies.json') -> List[Dict]:
        """Load company data from JSON file"""
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Companies file {filename} not found")
            return []

    def load_market_keywords(self, filename: str = 'market.json') -> Set[str]:
        """Load market keywords from market.json for intelligent page filtering"""
        try:
            with open(filename, 'r') as f:
                market_data = json.load(f)

            keywords = set()

            # Handle flat dictionary format: {"Category": ["item1", "item2", ...]}
            if isinstance(market_data, dict) and 'BusinessStructure' not in market_data:
                for category_name, items in market_data.items():
                    keywords.add(category_name.lower())
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, str):
                                keywords.add(item.lower())
            else:
                # Handle nested BusinessStructure format
                for business_line in market_data.get('BusinessStructure', []):
                    bl_name = business_line.get('name', '').lower()
                    keywords.add(bl_name)
                    for sub_bl in business_line.get('sub_business_lines', []):
                        sub_name = sub_bl.get('name', '').lower()
                        keywords.add(sub_name)
                        for category in sub_bl.get('categories', []):
                            cat_name = category.get('name', '').lower()
                            keywords.add(cat_name)
                            for item in category.get('items', []):
                                keywords.add(item.lower())

            # Clean up keywords and create search-friendly versions
            cleaned_keywords = set()
            for keyword in keywords:
                if keyword and len(keyword) > 2:  # Skip very short terms
                    cleaned_keywords.add(keyword)
                    # Add individual words for better matching
                    words = keyword.replace('-', ' ').replace('/', ' ').split()
                    for word in words:
                        if len(word) > 3:  # Only significant words
                            cleaned_keywords.add(word)

            # Add core oil & gas industry terms that are essential for link discovery
            core_industry_keywords = {
                'oil', 'gas', 'crude', 'petroleum', 'energy', 'fuel', 'power',
                'exploration', 'production', 'drilling', 'extraction', 'reservoir',
                'upstream', 'downstream', 'midstream', 'offshore', 'onshore',
                'refinery', 'refining', 'petrochemical', 'chemical', 'polymer',
                'lng', 'liquefied', 'natural gas', 'pipeline', 'terminal',
                'renewable', 'solar', 'wind', 'hydrogen', 'carbon', 'emissions',
                'sustainability', 'climate', 'decarbonization', 'net zero',
                'ccus', 'carbon capture', 'storage',
                'expertise', 'activities', 'operations', 'segments', 'divisions',
                'explore', 'produce', 'transform', 'supply', 'distribute',
            }
            cleaned_keywords.update(core_industry_keywords)

            logger.info(f"Loaded {len(cleaned_keywords)} market keywords for intelligent page filtering")
            return cleaned_keywords

        except FileNotFoundError:
            logger.warning("market.json not found, using default keywords")
            return set(['technology', 'innovation', 'market', 'business', 'energy'])
        except Exception as e:
            logger.error(f"Error loading market keywords: {str(e)}")
            return set(['technology', 'innovation', 'market', 'business', 'energy'])

    def get_robots_parser(self, base_url: str) -> Optional[RobotFileParser]:
        """Fetch and parse robots.txt for a domain"""
        parsed = urlparse(base_url)
        domain = f"{parsed.scheme}://{parsed.netloc}"

        if domain in self.robots_cache:
            return self.robots_cache[domain]

        robots_url = f"{domain}/robots.txt"
        logger.info(f"[ROBOTS] Checking robots.txt at {robots_url}")

        try:
            response = self._robots_session.get(robots_url, timeout=10, verify=self.verify_ssl)
            rp = RobotFileParser()
            rp.set_url(robots_url)

            if response.status_code == 200:
                rp.parse(response.text.splitlines())
                logger.info(f"[ROBOTS] Successfully parsed robots.txt for {domain}")

                crawl_delay = rp.crawl_delay('*')
                if crawl_delay:
                    logger.info(f"[ROBOTS] Crawl delay for {domain}: {crawl_delay} seconds")
            else:
                logger.info(f"[ROBOTS] No robots.txt found for {domain} (status {response.status_code}), assuming allowed")
                rp.parse([])

            self.robots_cache[domain] = rp
            return rp

        except Exception as e:
            logger.warning(f"[ROBOTS] Could not fetch robots.txt for {domain}: {str(e)}")
            rp = RobotFileParser()
            rp.parse([])
            self.robots_cache[domain] = rp
            return rp

    def is_url_allowed(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt"""
        rp = self.get_robots_parser(url)
        if rp is None:
            return True

        # Use a generic user-agent for robots.txt checks
        allowed = rp.can_fetch('*', url)

        if not allowed:
            logger.warning(f"[ROBOTS] URL blocked by robots.txt: {url}")
            self.blocked_sites.append({
                'url': url,
                'reason': 'Blocked by robots.txt',
                'suggestion': 'Visit this page manually in a browser'
            })

        return allowed

    def get_crawl_delay(self, url: str) -> float:
        """Get crawl delay from robots.txt, default to 1 second"""
        rp = self.get_robots_parser(url)
        if rp:
            delay = rp.crawl_delay('*')
            if delay:
                return max(delay, 1.0)
        return 1.0

    def _extract_content_from_html(self, html: str) -> str:
        """Parse HTML and extract clean text content using BeautifulSoup."""
        soup = BeautifulSoup(html, 'html.parser')

        # Remove script, style, nav, footer, header noise
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()

        # Try semantic content selectors first
        content_selectors = [
            'main', '[role="main"]', '.main-content', '.content',
            '.article', 'article', '.page-content', '#content',
            '.post-content', '.entry-content',
        ]

        content = ""
        for selector in content_selectors:
            element = soup.select_one(selector)
            if element:
                content = element.get_text(separator=' ', strip=True)
                break

        if not content:
            content = soup.get_text(separator=' ', strip=True)

        # Clean up whitespace
        content = re.sub(r'\s+', ' ', content)
        return content[:8000]  # Increased limit for richer analysis

    def _extract_title_from_html(self, html: str) -> str:
        """Extract the page title from HTML."""
        soup = BeautifulSoup(html, 'html.parser')
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text(strip=True)

        # Fallback: look for h1
        h1 = soup.find('h1')
        if h1:
            return h1.get_text(strip=True)

        return ""

    def _find_relevant_links_from_html(self, base_url: str, html: str) -> List[str]:
        """Find relevant links from rendered HTML."""
        soup = BeautifulSoup(html, 'html.parser')
        return self.find_relevant_links(base_url, soup)

    def find_relevant_links(self, base_url: str, soup: BeautifulSoup) -> List[str]:
        """Find links relevant to market analysis by classifying link text and URL.

        Instead of matching against a huge keyword set (which matches almost
        everything on an oil-company website), this method:
          1. Rejects links whose text/URL clearly indicates irrelevant pages
             (careers, HR, legal, cookies, diversity, etc.)
          2. Scores links positively when the link text or URL path contains
             topic indicators for: technology, market/strategy, investment,
             partnership, news/press releases, innovation, operations,
             sustainability, and specific market.json terms.
        """

        # --- Exclusion patterns (link text OR url path) ---
        # These are topics that are never useful for market/technology analysis.
        exclude_patterns = [
            # Careers / HR
            'career', 'careers', 'job', 'jobs', 'vacancy', 'vacancies',
            'work with us', 'work for us', 'join us', 'join our team',
            'recruitment', 'hiring', 'talent', 'graduate programme',
            'graduate program', 'internship', 'apprentice',
            # Human rights / diversity / inclusion
            'human rights', 'diversity', 'inclusion', 'equal opportunity',
            'workplace culture', 'employee resource',
            # Legal / compliance / privacy
            'privacy', 'cookie', 'cookies', 'legal', 'terms of use',
            'terms and conditions', 'disclaimer', 'compliance',
            'data protection', 'gdpr', 'accessibility statement',
            # Authentication / utility
            'login', 'log in', 'sign in', 'register', 'my account',
            'search', 'sitemap', 'site map', 'contact us', 'contact',
            'subscribe', 'newsletter', 'rss', 'podcast',
            # Social / sharing
            'share on', 'follow us', 'social media',
            'facebook', 'twitter', 'linkedin', 'instagram', 'youtube',
            # Procurement / suppliers
            'supplier', 'procurement', 'tender', 'bid',
        ]

        # --- Positive topic indicators (link text OR url path) ---
        # Grouped by topic with weights. Higher weight = more relevant.
        # Focused on oil & energy market segments (no technology/AI/IT/digital)
        topic_indicators = {
            # Core market / strategy pages
            'strategy': (['strategy', 'strategic', 'our strategy',
                          'our business', 'what we do', 'who we are',
                          'business overview', 'our approach',
                          'segments', 'divisions', 'business areas',
                          'portfolio', 'sectors', 'industries',
                          'our operations', 'operations', 'business units'], 5),
            # Investment / financial
            'investment': (['investor', 'investors', 'investment',
                            'invest', 'annual report', 'financial',
                            'quarterly results', 'earnings',
                            'capital', 'shareholder'], 4),
            # Partnerships / M&A / alliances
            'partnership': (['partnership', 'partners', 'partner',
                             'joint venture', 'joint-venture',
                             'acquisition', 'alliance', 'collaboration',
                             'agreement', 'mou', 'memorandum',
                             'venture', 'ventures'], 5),
            # News / press / announcements
            'news': (['news', 'press', 'media', 'announcement',
                       'press release', 'press-release', 'newsroom',
                       'media centre', 'media center', 'latest',
                       'updates', 'stories'], 3),
            # Sustainability / energy transition
            'sustainability': (['sustainability', 'sustainable',
                                 'climate', 'carbon', 'net zero',
                                 'net-zero', 'emissions', 'clean energy',
                                 'energy transition', 'decarbonization',
                                 'decarbonisation', 'renewable',
                                 'renewables', 'esg',
                                 'environment', 'environmental'], 4),
            # Specific energy / oil-gas operations
            'operations': (['upstream', 'downstream', 'midstream',
                            'exploration', 'production', 'refining',
                            'refinery', 'petrochemical', 'chemicals',
                            'lng', 'natural gas', 'gas', 'oil',
                            'offshore', 'onshore', 'deepwater',
                            'subsea', 'drilling', 'well',
                            'crude', 'petroleum', 'hydrocarbon'], 5),
            # New energy / future markets
            'new_energy': (['hydrogen', 'solar', 'wind', 'biofuel',
                             'biofuels', 'battery', 'storage',
                             'electric', 'ev charging', 'fuel cell',
                             'geothermal', 'nuclear', 'ammonia',
                             'ccus', 'carbon capture', 'ccs'], 5),
            # Projects / growth
            'projects': (['project', 'projects', 'growth', 'expansion',
                           'development', 'major projects',
                           'key projects', 'new projects'], 3),
            # Market-specific pages
            'markets': (['market', 'markets', 'market segment',
                          'fertilizer', 'mining', 'metals',
                          'recycling', 'circular economy'], 4),
        }

        links = set()
        link_scores = {}

        for link in soup.find_all('a', href=True):
            href = link.get('href', '').lower()
            text = link.get_text().lower().strip()
            title_attr = link.get('title', '').lower()

            # Decode URL and resolve relative paths
            href_decoded = unquote(link['href'])
            full_url = urljoin(base_url, href_decoded)

            # Only include same-domain links
            if urlparse(full_url).netloc != urlparse(base_url).netloc:
                continue

            # Skip anchors, javascript, and file downloads
            if href.startswith('#') or href.startswith('javascript:'):
                continue
            if any(href.endswith(ext) for ext in ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.zip']):
                continue

            # Normalize URL path for matching
            url_path = urlparse(full_url).path.lower()
            normalized_path = url_path.replace('-', ' ').replace('/', ' ').replace('_', ' ')

            # Combined text to classify the link
            link_text = f"{text} {title_attr} {normalized_path}"

            # --- Step 1: Reject if link clearly matches an excluded topic ---
            excluded = False
            for pattern in exclude_patterns:
                if pattern in link_text:
                    excluded = True
                    break
            if excluded:
                continue

            # --- Step 2: Score positively by topic relevance ---
            score = 0
            for topic, (indicators, weight) in topic_indicators.items():
                for indicator in indicators:
                    if indicator in link_text:
                        score += weight
                        break  # one match per topic is enough

            # --- Step 3: Bonus for market.json-specific terms in link text ---
            # Only give a small bonus (these are very broad on oil company sites)
            for keyword in self.market_keywords:
                if len(keyword) > 5 and keyword in link_text:
                    score += 1
                    break  # cap market.json bonus at 1

            if score > 0:
                links.add(full_url)
                link_scores[full_url] = max(link_scores.get(full_url, 0), score)

        # Sort by relevance score descending
        sorted_links = sorted(links, key=lambda x: link_scores.get(x, 0), reverse=True)

        logger.info(f"Found {len(sorted_links)} relevant links (showing top 30)")
        if sorted_links[:5]:
            for lnk in sorted_links[:5]:
                logger.info(f"  Top link ({link_scores.get(lnk, 0)} pts): {lnk}")
        return sorted_links[:30]

    def scrape_page(self, url: str, company_name: str = "") -> str:
        """
        Scrape content from a single page using the enhanced 3-layer strategy.

        Layer 1: requests with randomized headers + circuit breaker
        Layer 2: Playwright (if WAF detected or JS needed)
        Layer 3: Wayback Machine (if all else fails)
        """
        logger.info(f"[URL VISIT] Visiting: {url}")

        # Check robots.txt first
        if not self.is_url_allowed(url):
            self.visited_urls.append({
                'url': url,
                'company': company_name,
                'status': 'blocked',
                'status_code': None,
                'error': 'Blocked by robots.txt'
            })
            return ""

        # Use the enhanced engine (handles all 12 challenges)
        html = self.engine.smart_fetch(url, company_name=company_name)

        if html is None:
            self.visited_urls.append({
                'url': url,
                'company': company_name,
                'status': 'error',
                'status_code': None,
                'error': 'Failed after all strategies (requests/Playwright/Wayback)'
            })
            return ""

        # Track success
        self.visited_urls.append({
            'url': url,
            'company': company_name,
            'status': 'success',
            'status_code': 200,
            'error': None,
            'fetch_method': self._get_last_fetch_method(),
        })
        logger.info(f"[URL SUCCESS] {url}")

        # Extract clean text content from the rendered HTML
        content = self._extract_content_from_html(html)

        # P7: Skip if content is duplicate of already-scraped page
        if self.engine.url_normalizer.is_content_duplicate(content):
            logger.info(f"[DEDUP] Content duplicate detected, skipping: {url}")
            return ""

        return content

    def _get_last_fetch_method(self) -> str:
        """Determine which method was used for the last fetch."""
        stats = self.engine.stats
        if stats.get("wayback_used", 0) > 0:
            return "wayback"
        if stats.get("playwright_used", 0) > 0:
            return "playwright"
        return "requests"

    def categorize_page(self, url: str, title: str, content: str) -> str:
        """Categorize the page based on URL and content (market-focused)"""
        url_lower = url.lower()
        content_lower = (title + " " + content).lower()

        if any(term in url_lower or term in content_lower for term in ['market', 'business', 'sector', 'segment']):
            return 'market'
        elif any(term in url_lower or term in content_lower for term in ['upstream', 'downstream', 'refin', 'petrochemical', 'lng', 'drilling']):
            return 'operations'
        elif any(term in url_lower or term in content_lower for term in ['renewable', 'sustain', 'clean', 'carbon', 'net zero']):
            return 'sustainability'
        elif any(term in url_lower or term in content_lower for term in ['hydrogen', 'solar', 'wind', 'biofuel', 'ammonia', 'nuclear']):
            return 'new_energy'
        elif any(term in url_lower or term in content_lower for term in ['investor', 'annual report', 'financial', 'earning']):
            return 'financial'
        else:
            return 'general'

    def scan_company(self, company: Dict, max_depth: int = 4, max_discovery_requests: int = 25) -> List[ScrapedContent]:
        """Scan a single company's website with recursive multi-level crawling.

        Keeps crawling deeper (up to max_depth levels) as long as new relevant
        links are discovered on each level. Stops early if no new links are found
        or if max_discovery_requests is reached.

        Returns list of ScrapedContent for concurrent scraping compatibility.
        """
        company_name = company['name']
        main_url = company['url']
        results: List[ScrapedContent] = []

        # P11: Skip if already done in a previous run
        if self.engine.checkpoint and self.engine.checkpoint.is_company_done(company_name):
            logger.info(f"[CHECKPOINT] Skipping {company_name} (already completed)")
            return results

        logger.info(f"{'='*60}")
        logger.info(f"Scanning {company_name} at {main_url}")
        logger.info(f"{'='*60}")

        # Check robots.txt first
        if not self.is_url_allowed(main_url):
            logger.warning(f"[BLOCKED] {company_name} main page blocked by robots.txt")
            self.visited_urls.append({
                'url': main_url,
                'company': company_name,
                'status': 'blocked',
                'status_code': None,
                'error': 'Blocked by robots.txt - manual research required'
            })
            return results

        # Get crawl delay from robots.txt
        crawl_delay = self.get_crawl_delay(main_url)
        logger.info(f"[ROBOTS] Using crawl delay of {crawl_delay}s for {company_name}")

        try:
            # Fetch the main page (enhanced: tries requests -> Playwright -> Wayback)
            logger.info(f"[URL VISIT] Visiting main page: {main_url}")
            main_html = self.engine.smart_fetch(main_url, company_name=company_name)

            if main_html is None:
                self.visited_urls.append({
                    'url': main_url,
                    'company': company_name,
                    'status': 'error',
                    'status_code': None,
                    'error': 'Failed to access main page after all strategies'
                })
                logger.error(f"[BLOCKED] Could not access {company_name}")
                return results

            # Track main page visit
            self.visited_urls.append({
                'url': main_url,
                'company': company_name,
                'status': 'success',
                'status_code': 200,
                'error': None
            })
            logger.info(f"[URL SUCCESS] {main_url}")

            # Recursive multi-level link discovery
            all_discovered = set()       # All unique discovered URLs
            all_discovered_list = []     # Preserve insertion order
            crawled_for_links = set()    # URLs already crawled to extract links
            crawled_for_links.add(main_url)
            discovery_requests_used = 0  # Track total discovery HTTP requests

            # Level 1: Find relevant links from the main page
            level1_links = self._find_relevant_links_from_html(main_url, main_html)
            allowed_links = [link for link in level1_links if self.is_url_allowed(link)]
            logger.info(f"[DEPTH 1] Found {len(allowed_links)} relevant links for {company_name}")
            for link in allowed_links:
                if link not in all_discovered and link != main_url:
                    all_discovered.add(link)
                    all_discovered_list.append(link)

            # Level 2+ : keep crawling discovered pages for deeper links
            current_level_links = list(all_discovered)  # Links to crawl next
            depth = 1
            for depth in range(2, max_depth + 1):
                if not current_level_links or discovery_requests_used >= max_discovery_requests:
                    break

                # Pick candidates from the current level: prefer shallower paths (section hubs)
                candidates = sorted(current_level_links,
                                    key=lambda u: len(urlparse(u).path.rstrip('/').split('/')))
                # Limit how many pages we crawl per level (fewer as we go deeper)
                per_level_limit = max(3, 10 - (depth - 2) * 2)
                candidates = [c for c in candidates if c not in crawled_for_links][:per_level_limit]

                if not candidates:
                    break

                logger.info(f"[DEPTH {depth}] Crawling {len(candidates)} pages for deeper links")
                newly_found = []

                for nav_url in candidates:
                    if discovery_requests_used >= max_discovery_requests:
                        logger.info(f"[DEPTH {depth}] Reached discovery request limit ({max_discovery_requests})")
                        break

                    crawled_for_links.add(nav_url)
                    time.sleep(crawl_delay)
                    discovery_requests_used += 1

                    try:
                        nav_html = self.engine.smart_fetch(nav_url, company_name=company_name)
                        if nav_html:
                            deeper_links = self._find_relevant_links_from_html(main_url, nav_html)
                            for link in deeper_links:
                                if (link not in all_discovered
                                        and link != main_url
                                        and self.is_url_allowed(link)):
                                    all_discovered.add(link)
                                    all_discovered_list.append(link)
                                    newly_found.append(link)
                    except Exception as e:
                        logger.warning(f"[DEPTH {depth}] Error crawling {nav_url}: {str(e)}")

                if newly_found:
                    logger.info(f"[DEPTH {depth}] Discovered {len(newly_found)} new links")
                else:
                    logger.info(f"[DEPTH {depth}] No new links found, stopping deeper crawl")
                    break

                # Next level will crawl the newly found links
                current_level_links = newly_found

            logger.info(f"Total unique relevant links for {company_name}: {len(all_discovered_list)} "
                         f"(used {discovery_requests_used} discovery requests across {min(depth, max_depth)} levels)")

            # Scrape each relevant page (limit total pages per company)
            scraped_urls = set()
            max_pages = 30
            for link in all_discovered_list[:max_pages]:
                if link in scraped_urls:
                    continue
                scraped_urls.add(link)

                # Respect crawl delay + add jitter to look human
                delay = crawl_delay + random.uniform(0.5, 1.5)
                time.sleep(delay)
                content = self.scrape_page(link, company_name)

                if content:
                    # Extract title from a fresh fetch (or from cached HTML)
                    title = ""
                    html = self.engine.smart_fetch(link, company_name=company_name)
                    if html:
                        title = self._extract_title_from_html(html)

                    page_type = self.categorize_page(link, title, content)

                    scraped_data = ScrapedContent(
                        company=company_name,
                        url=link,
                        title=title,
                        content=content,
                        page_type=page_type
                    )

                    results.append(scraped_data)
                    self.scraped_content.append(scraped_data)
                    logger.info(f"Scraped {page_type} page: {title[:50]}...")

            # P11: Mark company as done and save checkpoint
            if self.engine.checkpoint:
                self.engine.checkpoint.mark_company_done(company_name)
                self.engine.save_checkpoint()

        except Exception as e:
            error_msg = str(e)
            self.visited_urls.append({
                'url': main_url,
                'company': company_name,
                'status': 'error',
                'status_code': None,
                'error': error_msg
            })
            logger.error(f"[URL ERROR] {main_url} - Error: {error_msg}")
            logger.error(f"Error scanning {company_name}: {error_msg}")

        return results

    def analyze_with_ai(self, company_data: List[ScrapedContent]) -> CompanyAnalysis:
        """Analyze scraped content - market-focused analysis (placeholder for now)"""
        company_name = company_data[0].company if company_data else "Unknown"

        market_content = []

        for data in company_data:
            if data.page_type in ['market', 'operations', 'new_energy', 'sustainability']:
                market_content.append(data.content)

        all_content = " ".join([data.content for data in company_data])

        market_keywords = {
            'upstream': ['exploration', 'drilling', 'production', 'extraction'],
            'downstream': ['refining', 'petrochemical', 'retail', 'marketing'],
            'renewable': ['solar', 'wind', 'renewable', 'clean energy', 'hydrogen'],
            'gas': ['natural gas', 'lng', 'pipeline', 'gas distribution'],
            'chemicals': ['chemicals', 'petrochemicals', 'plastics', 'polymers'],
            'carbon_management': ['carbon capture', 'ccus', 'carbon storage', 'net zero'],
            'hydrogen': ['hydrogen', 'green hydrogen', 'blue hydrogen', 'ammonia'],
            'biofuels': ['biofuel', 'biodiesel', 'sustainable aviation fuel', 'renewable diesel'],
        }

        market_presence = []
        for market, keywords in market_keywords.items():
            if any(keyword in all_content.lower() for keyword in keywords):
                market_presence.append(market)

        return CompanyAnalysis(
            company=company_name,
            market_presence=", ".join(market_presence) if market_presence else "Not clearly identified",
            technologies="N/A (market-focused analysis)",
            innovations="N/A (market-focused analysis)",
            summary=f"{company_name} operates in {len(market_presence)} identified market segments."
        )

    def run_analysis(self) -> None:
        """Run the complete analysis process with enhanced scraping."""
        logger.info("Starting oil company analysis (enhanced scraping engine)...")

        companies = self.load_companies()
        if not companies:
            logger.error("No companies to analyze")
            return

        # P11: Try to resume from checkpoint
        if self._resume:
            if self.engine.load_checkpoint():
                logger.info("[CHECKPOINT] Resuming from previous run")

        # P6: Scrape companies concurrently (respects per-domain rate limits)
        logger.info(f"Scanning {len(companies)} companies...")

        # Use sequential scanning for now to respect rate limits cleanly
        # (concurrent mode available via self.engine.concurrent for advanced use)
        for company in companies:
            self.scan_company(company)
            time.sleep(2)  # Pause between companies

        # Group scraped content by company
        companies_data = {}
        for content in self.scraped_content:
            if content.company not in companies_data:
                companies_data[content.company] = []
            companies_data[content.company].append(content)

        # Analyze each company's data
        for company_name, company_content in companies_data.items():
            analysis = self.analyze_with_ai(company_content)
            self.analysis_results.append(analysis)

        # Save results
        self.save_results()

        # Log enhanced engine statistics
        stats = self.engine.get_stats()
        logger.info("=" * 60)
        logger.info("[ENHANCED ENGINE STATS]")
        logger.info(f"  Total requests:      {stats['total_requests']}")
        logger.info(f"  Successful (requests): {stats['requests_success']}")
        logger.info(f"  Playwright fallbacks:  {stats['playwright_used']}")
        logger.info(f"  Wayback fallbacks:     {stats['wayback_used']}")
        logger.info(f"  WAF/CAPTCHA detected:  {stats['waf_detected']}")
        logger.info(f"  Duplicates skipped:    {stats['duplicates_skipped']}")
        logger.info(f"  Total failed:          {stats['requests_failed']}")
        logger.info("=" * 60)

        # Clean up
        self.engine.close()

    def save_results(self) -> None:
        """Save analysis results to files"""
        # Save raw scraped content
        scraped_data = [asdict(content) for content in self.scraped_content]
        with open('scraped_content.json', 'w', encoding='utf-8') as f:
            json.dump(scraped_data, f, indent=2, ensure_ascii=False)

        # Save analysis results
        analysis_data = [asdict(analysis) for analysis in self.analysis_results]
        with open('company_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)

        # Save visited URLs log
        with open('visited_urls.json', 'w', encoding='utf-8') as f:
            json.dump(self.visited_urls, f, indent=2, ensure_ascii=False)

        # Log URL visit summary
        success_count = sum(1 for url in self.visited_urls if url['status'] == 'success')
        error_count = sum(1 for url in self.visited_urls if url['status'] == 'error')
        blocked_count = sum(1 for url in self.visited_urls if url['status'] == 'blocked')
        logger.info(f"[URL SUMMARY] Total URLs: {len(self.visited_urls)}, Success: {success_count}, Errors: {error_count}, Blocked: {blocked_count}")

        # Save blocked sites
        if self.blocked_sites:
            with open('blocked_sites.json', 'w', encoding='utf-8') as f:
                json.dump(self.blocked_sites, f, indent=2, ensure_ascii=False)
            logger.info(f"[BLOCKED SITES] {len(self.blocked_sites)} sites require manual research - see blocked_sites.json")

        # Save enhanced engine statistics
        stats = self.engine.get_stats()
        with open('scraping_stats.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False, default=str)

        # Log errors and blocked sites
        if error_count > 0 or blocked_count > 0:
            logger.warning("=" * 60)
            logger.warning("[MANUAL RESEARCH REQUIRED] The following sites need manual checking:")
            for url_info in self.visited_urls:
                if url_info['status'] in ['error', 'blocked']:
                    logger.warning(f"  - {url_info['url']} ({url_info['company']})")
                    logger.warning(f"    Reason: {url_info['error']}")

            if self.blocked_sites:
                logger.warning("")
                logger.warning("[SUGGESTIONS FOR BLOCKED SITES]:")
                for site in self.blocked_sites:
                    logger.warning(f"  - {site['url']}")
                    logger.warning(f"    Suggestion: {site['suggestion']}")
            logger.warning("=" * 60)

        # Create a readable summary report
        with open('analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write("Oil Companies Market Analysis Report\n")
            f.write("=" * 50 + "\n\n")

            for analysis in self.analysis_results:
                f.write(f"Company: {analysis.company}\n")
                f.write(f"Market Presence: {analysis.market_presence}\n")
                f.write(f"Technologies: {analysis.technologies}\n")
                f.write(f"Innovations: {analysis.innovations}\n")
                f.write(f"Summary: {analysis.summary}\n")
                f.write("-" * 40 + "\n\n")

        logger.info("Results saved to scraped_content.json, company_analysis.json, visited_urls.json, scraping_stats.json, and analysis_report.txt")

if __name__ == "__main__":
    scanner = OilCompanyScanner()
    scanner.run_analysis()
