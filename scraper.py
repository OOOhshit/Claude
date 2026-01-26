#!/usr/bin/env python3
"""
Oil Companies Market & Technology Scanner

This script scans oil company websites to identify their market presence and technologies.
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
    def __init__(self):
        self.session = requests.Session()
        # Use more realistic browser headers to avoid detection
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        })
        self.scraped_content: List[ScrapedContent] = []
        self.analysis_results: List[CompanyAnalysis] = []
        self.market_keywords = self.load_market_keywords()
        self.visited_urls: List[Dict] = []  # Track all visited URLs with status
        self.robots_cache: Dict[str, RobotFileParser] = {}  # Cache robots.txt parsers
        self.blocked_sites: List[Dict] = []  # Track sites that block scraping
        
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
            
            # Extract all market names and items from the business structure
            for business_line in market_data.get('BusinessStructure', []):
                # Add business line name
                bl_name = business_line.get('name', '').lower()
                keywords.add(bl_name)
                
                # Process sub business lines
                for sub_bl in business_line.get('sub_business_lines', []):
                    sub_name = sub_bl.get('name', '').lower()
                    keywords.add(sub_name)
                    
                    # Process categories
                    for category in sub_bl.get('categories', []):
                        cat_name = category.get('name', '').lower()
                        keywords.add(cat_name)
                        
                        # Process items
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
            response = self.session.get(robots_url, timeout=10)
            rp = RobotFileParser()
            rp.set_url(robots_url)

            if response.status_code == 200:
                rp.parse(response.text.splitlines())
                logger.info(f"[ROBOTS] Successfully parsed robots.txt for {domain}")

                # Check for crawl delay
                crawl_delay = rp.crawl_delay('*')
                if crawl_delay:
                    logger.info(f"[ROBOTS] Crawl delay for {domain}: {crawl_delay} seconds")
            else:
                # No robots.txt means everything is allowed
                logger.info(f"[ROBOTS] No robots.txt found for {domain} (status {response.status_code}), assuming allowed")
                rp.parse([])  # Empty rules = allow all

            self.robots_cache[domain] = rp
            return rp

        except Exception as e:
            logger.warning(f"[ROBOTS] Could not fetch robots.txt for {domain}: {str(e)}")
            # If we can't fetch robots.txt, create permissive parser
            rp = RobotFileParser()
            rp.parse([])
            self.robots_cache[domain] = rp
            return rp

    def is_url_allowed(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt"""
        rp = self.get_robots_parser(url)
        if rp is None:
            return True

        user_agent = self.session.headers.get('User-Agent', '*')
        allowed = rp.can_fetch(user_agent, url)

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
                return max(delay, 1.0)  # At least 1 second
        return 1.0  # Default delay

    def request_with_retry(self, url: str, max_retries: int = 3) -> Optional[requests.Response]:
        """Make HTTP request with retry logic and exponential backoff"""
        for attempt in range(max_retries):
            try:
                # Add small random delay to appear more human-like
                if attempt > 0:
                    wait_time = (2 ** attempt) + random.uniform(0.5, 1.5)
                    logger.info(f"[RETRY] Waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)

                response = self.session.get(url, timeout=20)

                # Check for blocking responses
                if response.status_code == 403:
                    logger.warning(f"[BLOCKED] Access forbidden (403) for {url}")
                    self.blocked_sites.append({
                        'url': url,
                        'reason': 'Access forbidden (403) - Site may block scrapers',
                        'suggestion': 'Visit manually or check if site requires login/cookies'
                    })
                    return None
                elif response.status_code == 429:
                    logger.warning(f"[RATE LIMITED] Too many requests (429) for {url}")
                    if attempt < max_retries - 1:
                        continue  # Retry with backoff
                    return None
                elif response.status_code >= 500:
                    logger.warning(f"[SERVER ERROR] Status {response.status_code} for {url}")
                    if attempt < max_retries - 1:
                        continue  # Retry
                    return None

                response.raise_for_status()
                return response

            except requests.exceptions.Timeout:
                logger.warning(f"[TIMEOUT] Request timed out for {url} (attempt {attempt + 1}/{max_retries})")
                if attempt == max_retries - 1:
                    self.blocked_sites.append({
                        'url': url,
                        'reason': 'Request timeout - Site may be slow or blocking',
                        'suggestion': 'Try visiting manually or check network connectivity'
                    })
            except requests.exceptions.SSLError as e:
                logger.error(f"[SSL ERROR] SSL certificate error for {url}: {str(e)}")
                self.blocked_sites.append({
                    'url': url,
                    'reason': f'SSL certificate error: {str(e)}',
                    'suggestion': 'Site may have certificate issues, try visiting in browser'
                })
                return None
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"[CONNECTION ERROR] Could not connect to {url}: {str(e)}")
                if attempt == max_retries - 1:
                    self.blocked_sites.append({
                        'url': url,
                        'reason': 'Connection failed - Site may be down or blocking',
                        'suggestion': 'Check if site is accessible in browser'
                    })
            except Exception as e:
                logger.error(f"[ERROR] Unexpected error for {url}: {str(e)}")
                if attempt == max_retries - 1:
                    return None

        return None

    def find_relevant_links(self, base_url: str, soup: BeautifulSoup) -> List[str]:
        """Find links related to markets from market.json and general business topics"""
        # Combine market keywords with general business keywords
        all_keywords = self.market_keywords.union({
            'technology', 'innovation', 'digital', 'research', 'development',
            'business', 'operations', 'sustainability', 'future', 'solutions',
            'projects', 'ventures', 'investments', 'strategy', 'portfolio'
        })
        
        links = set()
        link_scores = {}  # Track relevance scores for better ranking
        
        for link in soup.find_all('a', href=True):
            href = link.get('href', '').lower()
            text = link.get_text().lower().strip()
            title = link.get('title', '').lower()
            
            # Decode any pre-encoded URLs and join properly
            href_decoded = unquote(link['href'])
            full_url = urljoin(base_url, href_decoded)
            
            # Only include links from the same domain
            if urlparse(full_url).netloc != urlparse(base_url).netloc:
                continue
            
            # Skip common non-content links
            if any(skip in href for skip in ['login', 'search', 'contact', 'privacy', 'cookie', 'legal']):
                continue
            
            # Calculate relevance score
            score = 0
            content_to_check = f"{href} {text} {title}"
            
            for keyword in all_keywords:
                if keyword in content_to_check:
                    # Market-specific keywords get higher scores
                    if keyword in self.market_keywords:
                        score += 3
                    else:
                        score += 1
            
            if score > 0:
                links.add(full_url)
                link_scores[full_url] = score
        
        # Sort by relevance score and return top links
        sorted_links = sorted(links, key=lambda x: link_scores.get(x, 0), reverse=True)
        
        logger.info(f"Found {len(sorted_links)} relevant links (showing top 15)")
        return sorted_links[:15]  # Increased from 10 to 15 for better coverage
    
    def scrape_page(self, url: str, company_name: str = "") -> str:
        """Scrape content from a single page with robots.txt checking and retry logic"""
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

        try:
            # Use retry logic
            response = self.request_with_retry(url)

            if response is None:
                self.visited_urls.append({
                    'url': url,
                    'company': company_name,
                    'status': 'error',
                    'status_code': None,
                    'error': 'Failed after retries - site may block scrapers'
                })
                return ""

            # Track successful visit
            self.visited_urls.append({
                'url': url,
                'company': company_name,
                'status': 'success',
                'status_code': response.status_code,
                'error': None
            })
            logger.info(f"[URL SUCCESS] {url} - Status: {response.status_code}")

            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Extract main content
            content_selectors = [
                'main', '[role="main"]', '.main-content', '.content',
                '.article', 'article', '.page-content'
            ]

            content = ""
            for selector in content_selectors:
                element = soup.select_one(selector)
                if element:
                    content = element.get_text(strip=True)
                    break

            if not content:
                content = soup.get_text(strip=True)

            # Clean up the content
            content = re.sub(r'\s+', ' ', content)
            return content[:5000]  # Limit content length

        except Exception as e:
            error_msg = str(e)
            self.visited_urls.append({
                'url': url,
                'company': company_name,
                'status': 'error',
                'status_code': None,
                'error': error_msg
            })
            logger.error(f"[URL ERROR] {url} - Error: {error_msg}")
            return ""
    
    def categorize_page(self, url: str, title: str, content: str) -> str:
        """Categorize the page based on URL and content"""
        url_lower = url.lower()
        content_lower = (title + " " + content).lower()
        
        if any(term in url_lower or term in content_lower for term in ['technolog', 'digital', 'innovation']):
            return 'technology'
        elif any(term in url_lower or term in content_lower for term in ['market', 'business', 'sector']):
            return 'market'
        elif any(term in url_lower or term in content_lower for term in ['research', 'development', 'r&d']):
            return 'research'
        elif any(term in url_lower or term in content_lower for term in ['renewable', 'sustain', 'clean']):
            return 'sustainability'
        else:
            return 'general'
    
    def scan_company(self, company: Dict) -> None:
        """Scan a single company's website with robots.txt respect and retry logic"""
        company_name = company['name']
        main_url = company['url']
        logger.info(f"Scanning {company_name} at {main_url}")

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
            return

        # Get crawl delay from robots.txt
        crawl_delay = self.get_crawl_delay(main_url)
        logger.info(f"[ROBOTS] Using crawl delay of {crawl_delay}s for {company_name}")

        try:
            logger.info(f"[URL VISIT] Visiting main page: {main_url}")

            # Use retry logic for main page
            response = self.request_with_retry(main_url)

            if response is None:
                self.visited_urls.append({
                    'url': main_url,
                    'company': company_name,
                    'status': 'error',
                    'status_code': None,
                    'error': 'Failed to access main page - site may block scrapers'
                })
                logger.error(f"[BLOCKED] Could not access {company_name} - manual research recommended")
                return

            # Track main page visit
            self.visited_urls.append({
                'url': main_url,
                'company': company_name,
                'status': 'success',
                'status_code': response.status_code,
                'error': None
            })
            logger.info(f"[URL SUCCESS] {main_url} - Status: {response.status_code}")

            soup = BeautifulSoup(response.content, 'html.parser')

            # Find relevant links
            relevant_links = self.find_relevant_links(main_url, soup)
            logger.info(f"Found {len(relevant_links)} relevant links for {company_name}")

            # Filter out links blocked by robots.txt
            allowed_links = [link for link in relevant_links if self.is_url_allowed(link)]
            if len(allowed_links) < len(relevant_links):
                logger.info(f"[ROBOTS] {len(relevant_links) - len(allowed_links)} links blocked by robots.txt")

            # Scrape each relevant page
            for link in allowed_links:
                # Respect crawl delay
                time.sleep(crawl_delay)
                content = self.scrape_page(link, company_name)

                if content:
                    title = ""
                    try:
                        page_response = self.request_with_retry(link)
                        if page_response:
                            page_soup = BeautifulSoup(page_response.content, 'html.parser')
                            title_tag = page_soup.find('title')
                            title = title_tag.get_text() if title_tag else ""
                    except:
                        pass

                    page_type = self.categorize_page(link, title, content)

                    scraped_data = ScrapedContent(
                        company=company_name,
                        url=link,
                        title=title,
                        content=content,
                        page_type=page_type
                    )

                    self.scraped_content.append(scraped_data)
                    logger.info(f"Scraped {page_type} page: {title[:50]}...")

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
    
    def analyze_with_ai(self, company_data: List[ScrapedContent]) -> CompanyAnalysis:
        """Analyze scraped content using AI (placeholder for now)"""
        company_name = company_data[0].company if company_data else "Unknown"
        
        # Group content by type
        tech_content = []
        market_content = []
        innovation_content = []
        
        for data in company_data:
            if data.page_type == 'technology':
                tech_content.append(data.content)
            elif data.page_type == 'market':
                market_content.append(data.content)
            elif data.page_type in ['research', 'innovation']:
                innovation_content.append(data.content)
        
        # Simple analysis based on keyword frequency
        all_content = " ".join([data.content for data in company_data])
        
        # Market analysis
        market_keywords = {
            'upstream': ['exploration', 'drilling', 'production', 'extraction'],
            'downstream': ['refining', 'petrochemical', 'retail', 'marketing'],
            'renewable': ['solar', 'wind', 'renewable', 'clean energy', 'hydrogen'],
            'gas': ['natural gas', 'lng', 'pipeline', 'gas distribution'],
            'chemicals': ['chemicals', 'petrochemicals', 'plastics', 'polymers']
        }
        
        market_presence = []
        for market, keywords in market_keywords.items():
            if any(keyword in all_content.lower() for keyword in keywords):
                market_presence.append(market)
        
        # Technology analysis
        tech_keywords = [
            'artificial intelligence', 'ai', 'machine learning', 'digital',
            'automation', 'robotics', 'iot', 'blockchain', 'cloud',
            'carbon capture', 'ccus', 'subsea', 'deepwater'
        ]
        
        technologies = [tech for tech in tech_keywords if tech in all_content.lower()]
        
        return CompanyAnalysis(
            company=company_name,
            market_presence=", ".join(market_presence) if market_presence else "Not clearly identified",
            technologies=", ".join(technologies) if technologies else "Traditional oil & gas technologies",
            innovations="Various innovation initiatives mentioned" if innovation_content else "Limited innovation content found",
            summary=f"{company_name} operates in {len(market_presence)} identified market segments with {len(technologies)} technology areas mentioned."
        )
    
    def run_analysis(self) -> None:
        """Run the complete analysis process"""
        logger.info("Starting oil company analysis...")
        
        companies = self.load_companies()
        if not companies:
            logger.error("No companies to analyze")
            return
        
        # Scan each company
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

        # Save blocked sites with suggestions for manual research
        if self.blocked_sites:
            with open('blocked_sites.json', 'w', encoding='utf-8') as f:
                json.dump(self.blocked_sites, f, indent=2, ensure_ascii=False)
            logger.info(f"[BLOCKED SITES] {len(self.blocked_sites)} sites require manual research - see blocked_sites.json")

        # Log all errors and blocked sites
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
            f.write("Oil Companies Market & Technology Analysis Report\n")
            f.write("=" * 50 + "\n\n")

            for analysis in self.analysis_results:
                f.write(f"Company: {analysis.company}\n")
                f.write(f"Market Presence: {analysis.market_presence}\n")
                f.write(f"Technologies: {analysis.technologies}\n")
                f.write(f"Innovations: {analysis.innovations}\n")
                f.write(f"Summary: {analysis.summary}\n")
                f.write("-" * 40 + "\n\n")

        logger.info("Results saved to scraped_content.json, company_analysis.json, visited_urls.json, and analysis_report.txt")

if __name__ == "__main__":
    scanner = OilCompanyScanner()
    scanner.run_analysis()