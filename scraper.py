#!/usr/bin/env python3
"""
Oil Companies Market & Technology Scanner

This script scans oil company websites to identify their market presence and technologies.
"""

import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import logging
from typing import List, Dict, Set
import re
from dataclasses import dataclass, asdict

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
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.scraped_content: List[ScrapedContent] = []
        self.analysis_results: List[CompanyAnalysis] = []
        self.market_keywords = self.load_market_keywords()
        
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
            
            full_url = urljoin(base_url, link['href'])
            
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
    
    def scrape_page(self, url: str) -> str:
        """Scrape content from a single page"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
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
            logger.error(f"Error scraping {url}: {str(e)}")
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
        """Scan a single company's website"""
        logger.info(f"Scanning {company['name']} at {company['url']}")
        
        try:
            # Get the main page
            response = self.session.get(company['url'], timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find relevant links
            relevant_links = self.find_relevant_links(company['url'], soup)
            logger.info(f"Found {len(relevant_links)} relevant links for {company['name']}")
            
            # Scrape each relevant page
            for link in relevant_links:
                time.sleep(1)  # Be respectful with requests
                content = self.scrape_page(link)
                
                if content:
                    title = ""
                    try:
                        page_response = self.session.get(link, timeout=10)
                        page_soup = BeautifulSoup(page_response.content, 'html.parser')
                        title_tag = page_soup.find('title')
                        title = title_tag.get_text() if title_tag else ""
                    except:
                        pass
                    
                    page_type = self.categorize_page(link, title, content)
                    
                    scraped_data = ScrapedContent(
                        company=company['name'],
                        url=link,
                        title=title,
                        content=content,
                        page_type=page_type
                    )
                    
                    self.scraped_content.append(scraped_data)
                    logger.info(f"Scraped {page_type} page: {title[:50]}...")
            
        except Exception as e:
            logger.error(f"Error scanning {company['name']}: {str(e)}")
    
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
        
        logger.info("Results saved to scraped_content.json, company_analysis.json, and analysis_report.txt")

if __name__ == "__main__":
    scanner = OilCompanyScanner()
    scanner.run_analysis()