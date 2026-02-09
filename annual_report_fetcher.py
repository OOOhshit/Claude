#!/usr/bin/env python3
"""
Annual Report Fetcher Module

Searches for and downloads company annual reports, extracts text content
for analysis of market presence and technology focus areas.
"""

import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, quote_plus
import time
import logging
import re
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import io
import urllib3

# Try to import PDF processing libraries
try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

logger = logging.getLogger(__name__)


@dataclass
class AnnualReportContent:
    """Data class for annual report content"""
    company: str
    year: int
    title: str
    url: str
    content: str
    source_type: str  # 'pdf', 'html', 'investor_page'
    download_path: Optional[str]
    extraction_method: str
    page_count: int


class AnnualReportFetcher:
    """Fetches and processes annual reports from oil company websites"""

    def __init__(self, download_dir: str = 'annual_reports', verify_ssl: bool = True):
        self.session = requests.Session()
        self.verify_ssl = verify_ssl

        # Disable SSL warnings if verification is disabled (for corporate environments)
        if not verify_ssl:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            logger.warning("[SSL] SSL certificate verification is DISABLED for annual report fetcher")

        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/pdf',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        })
        self.download_dir = download_dir
        self.target_year = datetime.now().year - 1  # Default to last year (2025)
        self.annual_reports: List[AnnualReportContent] = []
        self.search_results: List[Dict] = []

        # Create download directory if it doesn't exist
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
            logger.info(f"Created annual reports directory: {download_dir}")

    def set_target_year(self, year: int):
        """Set the target year for annual report search"""
        self.target_year = year
        logger.info(f"Target year set to: {year}")

    def search_annual_report_urls(self, company: Dict) -> List[Dict]:
        """Search for annual report URLs on company website"""
        company_name = company['name']
        base_url = company['url']
        logger.info(f"[ANNUAL REPORT] Searching for {self.target_year} annual report for {company_name}")

        found_reports = []

        # Common investor relations and annual report URL patterns
        search_paths = [
            '/investors',
            '/investor-relations',
            '/investor',
            '/en/investors',
            '/en/investor-relations',
            '/annual-report',
            '/annual-reports',
            '/reports',
            '/publications',
            '/en/reports',
            '/en/annual-reports',
            '/financials',
            '/financial-reports',
            '/corporate/investors',
            '/about/investors',
            '/about-us/investors',
        ]

        # Keywords to identify annual reports
        report_keywords = [
            f'annual report {self.target_year}',
            f'annual-report-{self.target_year}',
            f'{self.target_year} annual report',
            f'{self.target_year}-annual-report',
            f'integrated report {self.target_year}',
            f'{self.target_year} integrated report',
            'annual report',
            'integrated report',
            'yearly report',
            'annual review',
        ]

        # Try to find investor relations page first
        for path in search_paths:
            try:
                url = urljoin(base_url, path)
                logger.info(f"[ANNUAL REPORT] Checking: {url}")

                response = self.session.get(url, timeout=15, verify=self.verify_ssl)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')

                    # Look for PDF links and report pages
                    links = soup.find_all('a', href=True)
                    for link in links:
                        href = link.get('href', '').lower()
                        text = link.get_text().lower().strip()
                        title = link.get('title', '').lower()

                        # Check if this looks like an annual report link
                        combined_text = f"{href} {text} {title}"

                        # Must have year reference or be clearly an annual report
                        has_target_year = str(self.target_year) in combined_text
                        is_annual_report = any(kw in combined_text for kw in ['annual report', 'integrated report', 'annual-report'])
                        is_pdf = href.endswith('.pdf')

                        if has_target_year and (is_annual_report or is_pdf):
                            full_url = urljoin(url, link['href'])

                            # Avoid duplicates
                            if not any(r['url'] == full_url for r in found_reports):
                                found_reports.append({
                                    'company': company_name,
                                    'url': full_url,
                                    'title': link.get_text().strip()[:200],
                                    'year': self.target_year,
                                    'is_pdf': is_pdf or '.pdf' in full_url.lower(),
                                    'source_page': url
                                })
                                logger.info(f"[ANNUAL REPORT] Found: {link.get_text().strip()[:50]}...")

                time.sleep(1)  # Respectful delay

            except Exception as e:
                logger.warning(f"[ANNUAL REPORT] Error checking {path}: {str(e)}")
                continue

        # If no reports found, try searching with Google-like approach on site
        if not found_reports:
            found_reports = self._deep_search_for_reports(company)

        self.search_results.extend(found_reports)
        return found_reports

    def _deep_search_for_reports(self, company: Dict) -> List[Dict]:
        """Perform deeper search for annual reports"""
        company_name = company['name']
        base_url = company['url']
        found_reports = []

        logger.info(f"[ANNUAL REPORT] Performing deep search for {company_name}")

        try:
            # Get main page and look for any links containing year and report keywords
            response = self.session.get(base_url, timeout=15, verify=self.verify_ssl)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')

                # Find all links
                for link in soup.find_all('a', href=True):
                    href = link.get('href', '')
                    text = link.get_text().lower()

                    # Check for annual report indicators
                    if str(self.target_year) in (href + text):
                        if any(kw in (href.lower() + text) for kw in ['report', 'annual', 'pdf', 'investor', 'download']):
                            full_url = urljoin(base_url, href)

                            if not any(r['url'] == full_url for r in found_reports):
                                found_reports.append({
                                    'company': company_name,
                                    'url': full_url,
                                    'title': link.get_text().strip()[:200] or f"{company_name} {self.target_year} Report",
                                    'year': self.target_year,
                                    'is_pdf': '.pdf' in full_url.lower(),
                                    'source_page': base_url
                                })

        except Exception as e:
            logger.warning(f"[ANNUAL REPORT] Deep search error for {company_name}: {str(e)}")

        return found_reports

    def download_pdf(self, url: str, company_name: str) -> Optional[str]:
        """Download a PDF file and return local path"""
        try:
            logger.info(f"[ANNUAL REPORT] Downloading PDF from: {url}")

            response = self.session.get(url, timeout=60, stream=True, verify=self.verify_ssl)
            if response.status_code == 200:
                # Create safe filename
                safe_company = re.sub(r'[^\w\-]', '_', company_name)
                filename = f"{safe_company}_{self.target_year}_annual_report.pdf"
                filepath = os.path.join(self.download_dir, filename)

                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                logger.info(f"[ANNUAL REPORT] Downloaded: {filepath}")
                return filepath
            else:
                logger.warning(f"[ANNUAL REPORT] Failed to download PDF: HTTP {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"[ANNUAL REPORT] PDF download error: {str(e)}")
            return None

    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, int, str]:
        """Extract text from PDF file. Returns (text, page_count, method)"""
        text = ""
        page_count = 0
        method = "none"

        # Try pdfplumber first (better extraction)
        if HAS_PDFPLUMBER:
            try:
                logger.info(f"[ANNUAL REPORT] Extracting text with pdfplumber: {pdf_path}")
                with pdfplumber.open(pdf_path) as pdf:
                    page_count = len(pdf.pages)
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"

                if text.strip():
                    method = "pdfplumber"
                    logger.info(f"[ANNUAL REPORT] Extracted {len(text)} characters from {page_count} pages")
                    return text, page_count, method
            except Exception as e:
                logger.warning(f"[ANNUAL REPORT] pdfplumber extraction failed: {str(e)}")

        # Fall back to PyPDF2
        if HAS_PYPDF2:
            try:
                logger.info(f"[ANNUAL REPORT] Extracting text with PyPDF2: {pdf_path}")
                with open(pdf_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    page_count = len(reader.pages)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"

                if text.strip():
                    method = "PyPDF2"
                    logger.info(f"[ANNUAL REPORT] Extracted {len(text)} characters from {page_count} pages")
                    return text, page_count, method
            except Exception as e:
                logger.warning(f"[ANNUAL REPORT] PyPDF2 extraction failed: {str(e)}")

        if not HAS_PYPDF2 and not HAS_PDFPLUMBER:
            logger.error("[ANNUAL REPORT] No PDF extraction library available. Install pdfplumber or PyPDF2.")

        return text, page_count, method

    def extract_text_from_pdf_url(self, url: str) -> Tuple[str, int, str]:
        """Extract text directly from PDF URL without saving to disk"""
        text = ""
        page_count = 0
        method = "none"

        try:
            logger.info(f"[ANNUAL REPORT] Fetching PDF from URL: {url}")
            response = self.session.get(url, timeout=60, verify=self.verify_ssl)

            if response.status_code != 200:
                logger.warning(f"[ANNUAL REPORT] Failed to fetch PDF: HTTP {response.status_code}")
                return text, page_count, method

            pdf_content = io.BytesIO(response.content)

            # Try pdfplumber first
            if HAS_PDFPLUMBER:
                try:
                    with pdfplumber.open(pdf_content) as pdf:
                        page_count = len(pdf.pages)
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"

                    if text.strip():
                        method = "pdfplumber"
                        return text, page_count, method
                except Exception as e:
                    logger.warning(f"[ANNUAL REPORT] pdfplumber URL extraction failed: {str(e)}")
                    pdf_content.seek(0)  # Reset for next attempt

            # Fall back to PyPDF2
            if HAS_PYPDF2:
                try:
                    reader = PyPDF2.PdfReader(pdf_content)
                    page_count = len(reader.pages)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"

                    if text.strip():
                        method = "PyPDF2"
                        return text, page_count, method
                except Exception as e:
                    logger.warning(f"[ANNUAL REPORT] PyPDF2 URL extraction failed: {str(e)}")

        except Exception as e:
            logger.error(f"[ANNUAL REPORT] PDF URL extraction error: {str(e)}")

        return text, page_count, method

    def fetch_html_report_content(self, url: str) -> Tuple[str, str]:
        """Fetch content from HTML annual report page"""
        try:
            logger.info(f"[ANNUAL REPORT] Fetching HTML content from: {url}")
            response = self.session.get(url, timeout=20, verify=self.verify_ssl)

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')

                # Remove scripts and styles
                for element in soup(['script', 'style', 'nav', 'header', 'footer']):
                    element.decompose()

                # Try to find main content
                content_selectors = [
                    'main', '[role="main"]', '.main-content', '.content',
                    '.article-content', 'article', '.report-content',
                    '.annual-report', '#content'
                ]

                text = ""
                for selector in content_selectors:
                    element = soup.select_one(selector)
                    if element:
                        text = element.get_text(separator=' ', strip=True)
                        break

                if not text:
                    text = soup.get_text(separator=' ', strip=True)

                # Clean up whitespace
                text = re.sub(r'\s+', ' ', text)

                # Get title
                title = soup.find('title')
                title_text = title.get_text() if title else f"Annual Report {self.target_year}"

                logger.info(f"[ANNUAL REPORT] Extracted {len(text)} characters from HTML")
                return text, title_text

        except Exception as e:
            logger.error(f"[ANNUAL REPORT] HTML fetch error: {str(e)}")

        return "", ""

    def process_company_reports(self, company: Dict, download_pdfs: bool = True) -> List[AnnualReportContent]:
        """Process all found annual reports for a company"""
        company_name = company['name']
        reports = []

        # Search for report URLs
        found_urls = self.search_annual_report_urls(company)

        if not found_urls:
            logger.warning(f"[ANNUAL REPORT] No {self.target_year} annual reports found for {company_name}")
            return reports

        for report_info in found_urls:
            url = report_info['url']
            is_pdf = report_info['is_pdf']

            content = ""
            download_path = None
            page_count = 0
            extraction_method = "none"
            source_type = "html"

            if is_pdf:
                source_type = "pdf"

                if download_pdfs:
                    # Download and extract from file
                    download_path = self.download_pdf(url, company_name)
                    if download_path:
                        content, page_count, extraction_method = self.extract_text_from_pdf(download_path)
                else:
                    # Extract directly from URL
                    content, page_count, extraction_method = self.extract_text_from_pdf_url(url)
            else:
                # HTML report page
                content, title = self.fetch_html_report_content(url)
                if title:
                    report_info['title'] = title
                page_count = 1
                extraction_method = "html_scrape"

            if content:
                # Limit content length for analysis (keep first 100K characters)
                if len(content) > 100000:
                    content = content[:100000] + "\n... [Content truncated]"

                report = AnnualReportContent(
                    company=company_name,
                    year=self.target_year,
                    title=report_info.get('title', f"{company_name} Annual Report {self.target_year}"),
                    url=url,
                    content=content,
                    source_type=source_type,
                    download_path=download_path,
                    extraction_method=extraction_method,
                    page_count=page_count
                )
                reports.append(report)
                self.annual_reports.append(report)

                logger.info(f"[ANNUAL REPORT] Successfully processed: {report.title[:50]}...")

        return reports

    def fetch_all_company_reports(self, companies: List[Dict], download_pdfs: bool = True) -> List[AnnualReportContent]:
        """Fetch annual reports for all companies"""
        logger.info(f"[ANNUAL REPORT] Starting annual report fetch for {len(companies)} companies (Year: {self.target_year})")

        all_reports = []

        for company in companies:
            reports = self.process_company_reports(company, download_pdfs)
            all_reports.extend(reports)

            # Delay between companies
            time.sleep(2)

        logger.info(f"[ANNUAL REPORT] Completed. Found {len(all_reports)} annual reports.")
        return all_reports

    def save_results(self, filename: str = 'annual_reports_content.json'):
        """Save extracted annual report content to JSON"""
        results = [asdict(report) for report in self.annual_reports]

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"[ANNUAL REPORT] Results saved to {filename}")

        # Also save search results
        search_file = 'annual_reports_search.json'
        with open(search_file, 'w', encoding='utf-8') as f:
            json.dump(self.search_results, f, indent=2, ensure_ascii=False)

        logger.info(f"[ANNUAL REPORT] Search results saved to {search_file}")

    def get_report_summary(self) -> Dict:
        """Get summary of fetched reports"""
        return {
            'total_reports': len(self.annual_reports),
            'target_year': self.target_year,
            'companies_with_reports': len(set(r.company for r in self.annual_reports)),
            'pdf_reports': len([r for r in self.annual_reports if r.source_type == 'pdf']),
            'html_reports': len([r for r in self.annual_reports if r.source_type == 'html']),
            'total_pages_processed': sum(r.page_count for r in self.annual_reports),
            'reports_by_company': {
                company: [r.title for r in self.annual_reports if r.company == company]
                for company in set(r.company for r in self.annual_reports)
            }
        }


if __name__ == "__main__":
    # Test the fetcher
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load companies
    with open('companies.json', 'r') as f:
        companies = json.load(f)

    # Initialize fetcher
    fetcher = AnnualReportFetcher()
    fetcher.set_target_year(2025)  # Current year annual report

    # Test with first company
    if companies:
        test_company = companies[0]
        reports = fetcher.process_company_reports(test_company, download_pdfs=True)

        print(f"\nFound {len(reports)} reports for {test_company['name']}")
        for report in reports:
            print(f"  - {report.title}")
            print(f"    URL: {report.url}")
            print(f"    Content length: {len(report.content)} chars")
            print(f"    Pages: {report.page_count}")
