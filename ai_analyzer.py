#!/usr/bin/env python3
"""
AI-powered analysis module for oil company market presence and technologies.
"""

import json
import re
from typing import List, Dict, Set, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class MarketSegment:
    name: str
    confidence: float
    evidence: List[str]

@dataclass
class Technology:
    name: str
    category: str
    confidence: float
    evidence: List[str]

@dataclass
class NewMarketOpportunity:
    name: str
    confidence: float
    evidence: List[str]
    potential_category: str  # Where this might fit in existing structure

@dataclass
class AnnualReportAnalysis:
    """Analysis results from annual report content"""
    company: str
    year: int
    strategic_priorities: List[Dict]  # name, description, confidence
    technology_investments: List[Dict]  # name, category, investment_signal, confidence
    market_expansions: List[Dict]  # market, region, confidence
    financial_highlights: Dict  # revenue_focus, capex_focus, investment_areas
    future_outlook: List[str]  # Key future focus areas mentioned
    key_projects: List[Dict]  # name, description, status
    partnerships: List[Dict]  # partner, focus_area
    risk_factors: List[str]  # Identified risk areas
    source_url: str


@dataclass
class CompanyProfile:
    company: str
    market_segments: List[MarketSegment]
    technologies: List[Technology]
    sustainability_focus: float
    innovation_score: float
    geographic_presence: List[str]
    new_market_opportunities: List[NewMarketOpportunity]
    annual_report_analysis: Optional[AnnualReportAnalysis] = None
    summary: str = ""

class AIAnalyzer:
    def __init__(self):
        self.known_markets = self.load_known_markets()
        self.market_keywords = {
            'upstream': {
                'keywords': ['exploration', 'drilling', 'production', 'extraction', 'reservoir', 'wellhead', 'offshore', 'onshore'],
                'weight': 1.0
            },
            'downstream': {
                'keywords': ['refining', 'petrochemical', 'retail', 'marketing', 'distribution', 'fuel', 'gasoline'],
                'weight': 1.0
            },
            'renewable_energy': {
                'keywords': ['solar', 'wind', 'renewable', 'clean energy', 'hydrogen', 'biofuel', 'biomass'],
                'weight': 1.2
            },
            'natural_gas': {
                'keywords': ['natural gas', 'lng', 'pipeline', 'gas distribution', 'methane'],
                'weight': 1.0
            },
            'chemicals': {
                'keywords': ['chemicals', 'petrochemicals', 'plastics', 'polymers', 'specialty chemicals'],
                'weight': 0.9
            },
            'carbon_management': {
                'keywords': ['carbon capture', 'ccus', 'carbon storage', 'carbon neutral', 'net zero'],
                'weight': 1.3
            }
        }
        
        self.technology_keywords = {
            'digital_technologies': {
                'keywords': ['artificial intelligence', 'ai', 'machine learning', 'digital twin', 'analytics'],
                'category': 'Digital'
            },
            'automation': {
                'keywords': ['automation', 'robotics', 'autonomous', 'remote operations'],
                'category': 'Automation'
            },
            'iot_sensors': {
                'keywords': ['iot', 'sensors', 'monitoring', 'smart systems', 'connected'],
                'category': 'IoT'
            },
            'subsea_technology': {
                'keywords': ['subsea', 'deepwater', 'underwater', 'subsea equipment'],
                'category': 'Subsea'
            },
            'carbon_tech': {
                'keywords': ['carbon capture', 'ccus', 'carbon utilization', 'co2 storage'],
                'category': 'Carbon Management'
            },
            'renewable_tech': {
                'keywords': ['solar panels', 'wind turbines', 'energy storage', 'battery', 'hydrogen production'],
                'category': 'Renewable'
            }
        }
        
        # Load additional keywords from market.json
        self.market_keywords.update(self.extract_market_segments_from_json())
        
        self.geographic_indicators = [
            'north america', 'usa', 'canada', 'mexico',
            'europe', 'uk', 'norway', 'netherlands', 'france',
            'middle east', 'saudi arabia', 'uae', 'qatar',
            'africa', 'nigeria', 'angola', 'egypt',
            'asia pacific', 'china', 'india', 'australia', 'singapore',
            'south america', 'brazil', 'argentina', 'venezuela'
        ]
    
    def load_known_markets(self, filename: str = 'market.json') -> Set[str]:
        """Load known markets from market.json for new market detection"""
        try:
            with open(filename, 'r') as f:
                market_data = json.load(f)
            
            known_markets = set()
            
            for business_line in market_data.get('BusinessStructure', []):
                for sub_bl in business_line.get('sub_business_lines', []):
                    for category in sub_bl.get('categories', []):
                        for item in category.get('items', []):
                            # Clean and normalize market names
                            clean_item = item.lower().strip()
                            known_markets.add(clean_item)
                            # Add variations
                            clean_item = clean_item.replace('/', ' ').replace('-', ' ')
                            known_markets.add(clean_item)
            
            logger.info(f"Loaded {len(known_markets)} known markets for new market detection")
            return known_markets
            
        except FileNotFoundError:
            logger.warning("market.json not found for new market detection")
            return set()
        except Exception as e:
            logger.error(f"Error loading known markets: {str(e)}")
            return set()
    
    def extract_market_segments_from_json(self) -> Dict[str, Dict]:
        """Extract market segments from market.json to enhance market detection"""
        try:
            with open('market.json', 'r') as f:
                market_data = json.load(f)
            
            segments = {}
            
            for business_line in market_data.get('BusinessStructure', []):
                bl_name = business_line.get('name', '')
                
                for sub_bl in business_line.get('sub_business_lines', []):
                    sub_name = sub_bl.get('name', '')
                    
                    for category in sub_bl.get('categories', []):
                        cat_name = category.get('name', '')
                        
                        # Create market segment from category
                        if cat_name and category.get('items'):
                            segment_key = cat_name.lower().replace(' ', '_').replace('&', 'and')
                            segments[segment_key] = {
                                'keywords': [item.lower() for item in category.get('items', [])],
                                'weight': 1.5,  # Higher weight for market.json terms
                                'business_line': bl_name,
                                'category': cat_name
                            }
            
            logger.info(f"Enhanced market detection with {len(segments)} segments from market.json")
            return segments
            
        except Exception as e:
            logger.error(f"Error extracting market segments from JSON: {str(e)}")
            return {}
    
    def extract_market_segments(self, content: str, url_content_map: Dict[str, str] = None) -> List[MarketSegment]:
        """Extract market segments from content with confidence scores and URL tracking"""
        segments = []
        content_lower = content.lower()

        for segment_name, segment_data in self.market_keywords.items():
            evidence = []
            keyword_count = 0

            for keyword in segment_data['keywords']:
                matches = len(re.findall(r'\b' + re.escape(keyword) + r'\b', content_lower))
                if matches > 0:
                    keyword_count += matches
                    # Find URLs where this keyword appears
                    if url_content_map:
                        urls_with_keyword = []
                        for url, page_content in url_content_map.items():
                            if re.search(r'\b' + re.escape(keyword) + r'\b', page_content.lower()):
                                urls_with_keyword.append(url)
                        if urls_with_keyword:
                            evidence.append({
                                'keyword': keyword,
                                'mention_count': matches,
                                'urls': urls_with_keyword
                            })
                    else:
                        evidence.append({
                            'keyword': keyword,
                            'mention_count': matches,
                            'urls': []
                        })

            if keyword_count > 0:
                # Calculate confidence based on keyword frequency and weight
                confidence = min(0.95, (keyword_count * segment_data['weight']) / 10)

                segments.append(MarketSegment(
                    name=segment_name.replace('_', ' ').title(),
                    confidence=confidence,
                    evidence=evidence[:5]  # Limit evidence items
                ))

        return sorted(segments, key=lambda x: x.confidence, reverse=True)
    
    def extract_technologies(self, content: str, url_content_map: Dict[str, str] = None) -> List[Technology]:
        """Extract technologies from content with categorization and URL tracking"""
        technologies = []
        content_lower = content.lower()

        for tech_name, tech_data in self.technology_keywords.items():
            evidence = []
            keyword_count = 0

            for keyword in tech_data['keywords']:
                matches = len(re.findall(r'\b' + re.escape(keyword) + r'\b', content_lower))
                if matches > 0:
                    keyword_count += matches
                    # Find URLs where this keyword appears
                    if url_content_map:
                        urls_with_keyword = []
                        for url, page_content in url_content_map.items():
                            if re.search(r'\b' + re.escape(keyword) + r'\b', page_content.lower()):
                                urls_with_keyword.append(url)
                        if urls_with_keyword:
                            evidence.append({
                                'keyword': keyword,
                                'mention_count': matches,
                                'urls': urls_with_keyword
                            })
                    else:
                        evidence.append({
                            'keyword': keyword,
                            'mention_count': matches,
                            'urls': []
                        })

            if keyword_count > 0:
                confidence = min(0.95, keyword_count / 5)

                technologies.append(Technology(
                    name=tech_name.replace('_', ' ').title(),
                    category=tech_data['category'],
                    confidence=confidence,
                    evidence=evidence[:3]
                ))

        return sorted(technologies, key=lambda x: x.confidence, reverse=True)
    
    def calculate_sustainability_focus(self, content: str) -> float:
        """Calculate sustainability focus score"""
        sustainability_keywords = [
            'sustainability', 'sustainable', 'renewable', 'clean energy',
            'carbon neutral', 'net zero', 'emissions reduction',
            'environmental', 'green', 'climate change'
        ]
        
        content_lower = content.lower()
        total_mentions = sum(len(re.findall(r'\b' + re.escape(keyword) + r'\b', content_lower)) 
                           for keyword in sustainability_keywords)
        
        # Normalize to 0-1 scale
        return min(1.0, total_mentions / 20)
    
    def calculate_innovation_score(self, content: str) -> float:
        """Calculate innovation score based on R&D and innovation mentions"""
        innovation_keywords = [
            'innovation', 'research', 'development', 'r&d',
            'breakthrough', 'cutting-edge', 'advanced',
            'pioneering', 'patent', 'technology development'
        ]
        
        content_lower = content.lower()
        total_mentions = sum(len(re.findall(r'\b' + re.escape(keyword) + r'\b', content_lower)) 
                           for keyword in innovation_keywords)
        
        return min(1.0, total_mentions / 15)
    
    def extract_geographic_presence(self, content: str) -> List[str]:
        """Extract geographic presence indicators"""
        content_lower = content.lower()
        present_regions = []
        
        for region in self.geographic_indicators:
            if region in content_lower:
                present_regions.append(region.title())
        
        return list(set(present_regions))
    
    def generate_completeness_suggestions(self, company_name: str, scraped_data: List[Dict], profile: CompanyProfile) -> Dict:
        """Generate suggestions to improve analysis completeness"""
        suggestions = {
            'company': company_name,
            'completeness_score': 0.0,
            'pages_analyzed': len(scraped_data),
            'suggestions': [],
            'missing_areas': [],
            'verification_steps': []
        }

        # Check page count
        if len(scraped_data) < 5:
            suggestions['suggestions'].append({
                'priority': 'HIGH',
                'issue': 'Low number of pages analyzed',
                'recommendation': f'Only {len(scraped_data)} pages were analyzed. Consider increasing the link limit or checking if the website has accessibility restrictions.'
            })

        # Check content diversity (page types)
        page_types = [item.get('page_type', 'general') for item in scraped_data]
        page_type_counts = {}
        for pt in page_types:
            page_type_counts[pt] = page_type_counts.get(pt, 0) + 1

        expected_types = ['technology', 'market', 'sustainability', 'research']
        missing_types = [t for t in expected_types if t not in page_type_counts]
        if missing_types:
            suggestions['missing_areas'].extend(missing_types)
            suggestions['suggestions'].append({
                'priority': 'MEDIUM',
                'issue': f'Missing content types: {", ".join(missing_types)}',
                'recommendation': f'Try searching the company website manually for pages related to: {", ".join(missing_types)}. These sections may use different URLs or require navigation through menus.'
            })

        # Check market segment coverage
        if len(profile.market_segments) < 2:
            suggestions['suggestions'].append({
                'priority': 'MEDIUM',
                'issue': 'Limited market segment detection',
                'recommendation': 'Check if the company has annual reports, investor presentations, or business unit pages that might contain more detailed market information.'
            })

        # Check technology coverage
        if len(profile.technologies) < 2:
            suggestions['suggestions'].append({
                'priority': 'MEDIUM',
                'issue': 'Limited technology detection',
                'recommendation': 'Look for R&D pages, innovation hubs, technology partnerships, or digital transformation sections on the website.'
            })

        # Check geographic presence
        if len(profile.geographic_presence) < 3:
            suggestions['suggestions'].append({
                'priority': 'LOW',
                'issue': 'Limited geographic information',
                'recommendation': 'Check for "Our Locations", "Global Operations", or "Where We Operate" sections that may list geographic presence.'
            })

        # Verification steps for thoroughness
        suggestions['verification_steps'] = [
            {
                'step': 'Manual Website Review',
                'description': 'Visit the company website and navigate through main menu items to identify any missed sections.',
                'urls_to_check': [
                    f'{scraped_data[0].get("url", "").split("/")[0]}//{scraped_data[0].get("url", "").split("/")[2]}/about' if scraped_data else '',
                    f'{scraped_data[0].get("url", "").split("/")[0]}//{scraped_data[0].get("url", "").split("/")[2]}/investors' if scraped_data else '',
                    f'{scraped_data[0].get("url", "").split("/")[0]}//{scraped_data[0].get("url", "").split("/")[2]}/sustainability' if scraped_data else '',
                ]
            },
            {
                'step': 'Check Annual Reports',
                'description': 'Annual reports often contain comprehensive market and technology information. Search for PDF downloads.',
                'search_terms': [f'{company_name} annual report', f'{company_name} investor relations']
            },
            {
                'step': 'Cross-Reference with News',
                'description': 'Recent news articles may reveal market activities not yet reflected on the corporate website.',
                'search_terms': [f'{company_name} technology', f'{company_name} new market', f'{company_name} expansion']
            },
            {
                'step': 'Check Regional Websites',
                'description': 'Companies often have regional websites with localized content that may provide additional insights.',
                'note': 'Look for language/region selectors on the main website'
            },
            {
                'step': 'Review Social Media & Press Releases',
                'description': 'Corporate LinkedIn, Twitter, and press release pages often announce new initiatives before website updates.',
                'search_terms': [f'{company_name} LinkedIn', f'{company_name} press releases']
            }
        ]

        # Calculate completeness score
        score = 0.0
        score += min(0.3, len(scraped_data) / 15 * 0.3)  # Up to 30% for page count
        score += min(0.2, len(page_type_counts) / 5 * 0.2)  # Up to 20% for content diversity
        score += min(0.2, len(profile.market_segments) / 5 * 0.2)  # Up to 20% for market segments
        score += min(0.15, len(profile.technologies) / 4 * 0.15)  # Up to 15% for technologies
        score += min(0.15, len(profile.geographic_presence) / 10 * 0.15)  # Up to 15% for geography

        suggestions['completeness_score'] = round(score, 2)

        # Add overall assessment
        if score < 0.4:
            suggestions['overall_assessment'] = 'LOW - Analysis may be incomplete. Review suggestions carefully.'
        elif score < 0.7:
            suggestions['overall_assessment'] = 'MEDIUM - Reasonable coverage but room for improvement.'
        else:
            suggestions['overall_assessment'] = 'HIGH - Good coverage of company information.'

        return suggestions

    def generate_summary(self, profile: CompanyProfile) -> str:
        """Generate a comprehensive summary including new market opportunities and annual report insights"""
        top_markets = [seg.name for seg in profile.market_segments[:3]]
        top_techs = [tech.name for tech in profile.technologies[:3]]

        summary = f"{profile.company} operates primarily in {', '.join(top_markets)} markets. "

        if top_techs:
            summary += f"Key technology focus areas include {', '.join(top_techs)}. "

        if profile.sustainability_focus > 0.5:
            summary += "The company shows strong commitment to sustainability and clean energy transition. "

        if profile.innovation_score > 0.5:
            summary += "High innovation activity with significant R&D investments. "

        if profile.new_market_opportunities:
            new_markets = [opp.name for opp in profile.new_market_opportunities[:2]]
            summary += f"[ALERT] EMERGING OPPORTUNITIES: Potential new markets identified including {', '.join(new_markets)}. "

        # Add annual report insights
        if profile.annual_report_analysis:
            ar = profile.annual_report_analysis
            if ar.strategic_priorities:
                top_priorities = [p['name'] for p in ar.strategic_priorities[:2]]
                summary += f"[ANNUAL REPORT {ar.year}] Strategic priorities: {', '.join(top_priorities)}. "

            if ar.technology_investments:
                strong_investments = [t['name'] for t in ar.technology_investments if t.get('investment_signal') == 'Strong'][:2]
                if strong_investments:
                    summary += f"Strong technology investments in: {', '.join(strong_investments)}. "

            if ar.market_expansions:
                expansions = [f"{e['market']} in {e['region']}" for e in ar.market_expansions[:2]]
                summary += f"Market expansion focus: {', '.join(expansions)}. "

        if len(profile.geographic_presence) > 5:
            summary += f"Global presence with operations in {len(profile.geographic_presence)} regions."

        return summary
    
    def detect_new_market_opportunities(self, content: str, url_content_map: Dict[str, str] = None) -> List[NewMarketOpportunity]:
        """Detect potentially new markets not in our known market list"""
        content_lower = content.lower()

        # Look for market indicators that might represent new opportunities
        new_market_patterns = [
            r'\b(\w+\s+(?:market|sector|industry|business|energy|fuel|technology))\b',
            r'\b(emerging\s+\w+)\b',
            r'\b(next-generation\s+\w+)\b',
            r'\b(new\s+\w+\s+(?:solutions|technologies|markets))\b',
            r'\b(\w+\s+transition)\b',
            r'\b(future\s+of\s+\w+)\b'
        ]

        potential_new_markets = set()

        for pattern in new_market_patterns:
            matches = re.findall(pattern, content_lower)
            for match in matches:
                clean_match = match.strip()
                # Check if this is not in our known markets
                if not any(known in clean_match or clean_match in known for known in self.known_markets):
                    if len(clean_match) > 5 and len(clean_match) < 50:  # Reasonable length
                        potential_new_markets.add(clean_match)

        # Convert to NewMarketOpportunity objects with evidence
        new_opportunities = []
        for market in potential_new_markets:
            # Find evidence in content
            market_mentions = content_lower.count(market)
            if market_mentions > 0:
                confidence = min(0.8, market_mentions / 5)  # Max 80% confidence

                # Try to categorize the new market
                potential_category = "Emerging Market"
                if any(term in market for term in ['energy', 'fuel', 'power']):
                    potential_category = "New Energy Solutions"
                elif any(term in market for term in ['technology', 'digital', 'ai']):
                    potential_category = "Technology Innovation"
                elif any(term in market for term in ['chemical', 'material']):
                    potential_category = "New Materials/Chemicals"

                # Find URLs where this market is mentioned
                urls_with_market = []
                if url_content_map:
                    for url, page_content in url_content_map.items():
                        if market in page_content.lower():
                            urls_with_market.append(url)

                evidence = [{
                    'keyword': market,
                    'mention_count': market_mentions,
                    'urls': urls_with_market
                }]

                new_opportunities.append(NewMarketOpportunity(
                    name=market.title(),
                    confidence=confidence,
                    evidence=evidence,
                    potential_category=potential_category
                ))

        # Sort by confidence
        return sorted(new_opportunities, key=lambda x: x.confidence, reverse=True)[:5]  # Top 5
    
    def analyze_company_content(self, company_name: str, scraped_data: List[Dict]) -> CompanyProfile:
        """Perform comprehensive AI analysis on company content"""
        logger.info(f"Analyzing content for {company_name}")

        # Build URL to content map for tracking keyword sources
        url_content_map = {}
        for item in scraped_data:
            url = item.get('url', '')
            content = item.get('content', '')
            if url and content:
                url_content_map[url] = content

        # Combine all content
        all_content = " ".join([item.get('content', '') for item in scraped_data])

        # Extract different aspects with URL tracking
        market_segments = self.extract_market_segments(all_content, url_content_map)
        technologies = self.extract_technologies(all_content, url_content_map)
        sustainability_focus = self.calculate_sustainability_focus(all_content)
        innovation_score = self.calculate_innovation_score(all_content)
        geographic_presence = self.extract_geographic_presence(all_content)
        new_market_opportunities = self.detect_new_market_opportunities(all_content, url_content_map)

        # Create profile
        profile = CompanyProfile(
            company=company_name,
            market_segments=market_segments,
            technologies=technologies,
            sustainability_focus=sustainability_focus,
            innovation_score=innovation_score,
            geographic_presence=geographic_presence,
            new_market_opportunities=new_market_opportunities,
            summary=""
        )

        # Generate summary
        profile.summary = self.generate_summary(profile)

        return profile
    
    def analyze_all_companies(self, scraped_content: List[Dict]) -> List[CompanyProfile]:
        """Analyze all companies from scraped content"""
        # Group content by company
        company_content = {}
        for item in scraped_content:
            company = item.get('company', 'Unknown')
            if company not in company_content:
                company_content[company] = []
            company_content[company].append(item)
        
        # Analyze each company
        profiles = []
        for company_name, content_list in company_content.items():
            profile = self.analyze_company_content(company_name, content_list)
            profiles.append(profile)
        
        return profiles
    
    def analyze_annual_report(self, company_name: str, year: int, content: str, source_url: str) -> AnnualReportAnalysis:
        """Analyze annual report content for strategic insights, technologies, and market focus"""
        logger.info(f"[ANNUAL REPORT ANALYSIS] Analyzing {year} annual report for {company_name}")

        content_lower = content.lower()

        # Extract strategic priorities
        strategic_priorities = self._extract_strategic_priorities(content)

        # Extract technology investments
        technology_investments = self._extract_technology_investments(content)

        # Extract market expansions
        market_expansions = self._extract_market_expansions(content)

        # Extract financial highlights
        financial_highlights = self._extract_financial_highlights(content)

        # Extract future outlook
        future_outlook = self._extract_future_outlook(content)

        # Extract key projects
        key_projects = self._extract_key_projects(content)

        # Extract partnerships
        partnerships = self._extract_partnerships(content)

        # Extract risk factors
        risk_factors = self._extract_risk_factors(content)

        return AnnualReportAnalysis(
            company=company_name,
            year=year,
            strategic_priorities=strategic_priorities,
            technology_investments=technology_investments,
            market_expansions=market_expansions,
            financial_highlights=financial_highlights,
            future_outlook=future_outlook,
            key_projects=key_projects,
            partnerships=partnerships,
            risk_factors=risk_factors,
            source_url=source_url
        )

    def _extract_strategic_priorities(self, content: str) -> List[Dict]:
        """Extract strategic priorities from annual report"""
        priorities = []
        content_lower = content.lower()

        # Strategic priority keywords and patterns
        strategy_patterns = [
            (r'strategic\s+(?:priority|priorities|focus|objective|pillar)[\s:]+([^.]+)', 'Strategic Priority'),
            (r'key\s+(?:priority|priorities|focus|objective)[\s:]+([^.]+)', 'Key Priority'),
            (r'our\s+strategy\s+(?:is|focuses|centers)\s+on\s+([^.]+)', 'Core Strategy'),
            (r'commitment\s+to\s+([^.]{10,100})', 'Commitment'),
            (r'(?:driving|accelerating|advancing)\s+([^.]{10,80})', 'Focus Area'),
        ]

        for pattern, priority_type in strategy_patterns:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            for match in matches[:5]:  # Limit matches
                clean_match = match.strip().title()
                if len(clean_match) > 10 and len(clean_match) < 200:
                    if not any(p['name'] == clean_match for p in priorities):
                        priorities.append({
                            'name': clean_match,
                            'type': priority_type,
                            'confidence': 0.7
                        })

        # Check for specific strategic themes
        strategic_themes = {
            'energy transition': 'Energy Transition',
            'net zero': 'Net Zero Emissions',
            'carbon neutral': 'Carbon Neutrality',
            'digital transformation': 'Digital Transformation',
            'operational excellence': 'Operational Excellence',
            'cost reduction': 'Cost Optimization',
            'portfolio optimization': 'Portfolio Optimization',
            'sustainable growth': 'Sustainable Growth',
            'renewable energy': 'Renewable Energy Expansion',
            'hydrogen': 'Hydrogen Development',
            'electric vehicle': 'EV Infrastructure',
            'circular economy': 'Circular Economy',
        }

        for theme, theme_name in strategic_themes.items():
            if theme in content_lower:
                count = content_lower.count(theme)
                if count >= 3:  # Must appear multiple times
                    if not any(p['name'] == theme_name for p in priorities):
                        priorities.append({
                            'name': theme_name,
                            'type': 'Strategic Theme',
                            'confidence': min(0.9, count / 10)
                        })

        return sorted(priorities, key=lambda x: x['confidence'], reverse=True)[:10]

    def _extract_technology_investments(self, content: str) -> List[Dict]:
        """Extract technology investment signals from annual report"""
        investments = []
        content_lower = content.lower()

        # Technology categories with investment signals
        tech_investment_keywords = {
            'Artificial Intelligence & Machine Learning': [
                'ai investment', 'machine learning', 'artificial intelligence',
                'predictive analytics', 'data science', 'neural network'
            ],
            'Digital Twin & Simulation': [
                'digital twin', 'simulation', 'virtual model', '3d modeling'
            ],
            'Cloud Computing': [
                'cloud computing', 'cloud platform', 'cloud infrastructure',
                'saas', 'digital platform'
            ],
            'Internet of Things (IoT)': [
                'iot', 'internet of things', 'connected devices', 'smart sensors',
                'industrial iot', 'iiot'
            ],
            'Automation & Robotics': [
                'automation', 'robotics', 'autonomous', 'unmanned',
                'robotic process', 'drone'
            ],
            'Carbon Capture (CCUS)': [
                'carbon capture', 'ccus', 'ccs', 'carbon storage',
                'direct air capture', 'co2 capture'
            ],
            'Hydrogen Technology': [
                'hydrogen production', 'green hydrogen', 'blue hydrogen',
                'hydrogen infrastructure', 'electrolysis', 'fuel cell'
            ],
            'Renewable Energy Tech': [
                'solar technology', 'wind technology', 'battery storage',
                'energy storage', 'offshore wind', 'solar panel'
            ],
            'Blockchain & Digital Ledger': [
                'blockchain', 'distributed ledger', 'smart contract'
            ],
            'Advanced Materials': [
                'advanced materials', 'nanomaterial', 'composite material',
                'lightweight material'
            ],
            'Biofuels & Sustainable Aviation': [
                'biofuel', 'sustainable aviation fuel', 'saf', 'renewable diesel',
                'hvo', 'biodiesel'
            ],
        }

        # Investment signal phrases
        investment_signals = [
            'invest', 'investment', 'investing', 'capex', 'capital expenditure',
            'spending', 'allocated', 'committed', 'deployed', 'funding',
            'development', 'expansion', 'partnership', 'acquisition'
        ]

        for tech_category, keywords in tech_investment_keywords.items():
            found_keywords = []
            has_investment_signal = False

            for keyword in keywords:
                if keyword in content_lower:
                    count = content_lower.count(keyword)
                    if count > 0:
                        found_keywords.append({'keyword': keyword, 'count': count})

                        # Check for nearby investment signals
                        for signal in investment_signals:
                            # Check within 200 characters of keyword mention
                            idx = content_lower.find(keyword)
                            while idx != -1:
                                context = content_lower[max(0, idx-100):idx+len(keyword)+100]
                                if signal in context:
                                    has_investment_signal = True
                                    break
                                idx = content_lower.find(keyword, idx + 1)

            if found_keywords:
                total_mentions = sum(k['count'] for k in found_keywords)
                confidence = min(0.9, total_mentions / 10)

                investments.append({
                    'name': tech_category,
                    'category': 'Technology',
                    'investment_signal': 'Strong' if has_investment_signal else 'Mentioned',
                    'evidence': found_keywords[:3],
                    'confidence': confidence if has_investment_signal else confidence * 0.6
                })

        return sorted(investments, key=lambda x: x['confidence'], reverse=True)

    def _extract_market_expansions(self, content: str) -> List[Dict]:
        """Extract market expansion signals from annual report"""
        expansions = []
        content_lower = content.lower()

        # Market expansion indicators
        expansion_signals = [
            'expansion', 'expanding', 'entering', 'entered', 'new market',
            'growth', 'growing', 'opportunity', 'development', 'launch',
            'investment in', 'presence in', 'operations in'
        ]

        # Geographic regions
        regions = {
            'North America': ['north america', 'united states', 'usa', 'canada', 'mexico', 'gulf of mexico'],
            'Europe': ['europe', 'european', 'uk', 'norway', 'netherlands', 'germany', 'north sea'],
            'Middle East': ['middle east', 'saudi arabia', 'uae', 'qatar', 'kuwait', 'oman', 'bahrain'],
            'Asia Pacific': ['asia', 'asia pacific', 'china', 'india', 'australia', 'indonesia', 'malaysia', 'singapore'],
            'Africa': ['africa', 'african', 'nigeria', 'angola', 'egypt', 'libya', 'mozambique'],
            'South America': ['south america', 'latin america', 'brazil', 'argentina', 'guyana', 'suriname'],
        }

        # Market segments
        market_segments = {
            'Upstream': ['exploration', 'drilling', 'production', 'upstream'],
            'Downstream': ['refining', 'petrochemical', 'retail', 'downstream'],
            'Renewable Energy': ['renewable', 'solar', 'wind', 'clean energy'],
            'Natural Gas & LNG': ['natural gas', 'lng', 'liquefied natural gas'],
            'Chemicals': ['chemicals', 'specialty chemicals', 'petrochemicals'],
            'Electric Vehicles': ['electric vehicle', 'ev charging', 'electrification'],
            'Hydrogen': ['hydrogen', 'green hydrogen', 'hydrogen hub'],
        }

        # Check for market-region combinations with expansion signals
        for market, market_kw in market_segments.items():
            for region, region_kw in regions.items():
                for m_kw in market_kw:
                    for r_kw in region_kw:
                        if m_kw in content_lower and r_kw in content_lower:
                            # Check for expansion signals nearby
                            for signal in expansion_signals:
                                pattern = f'{signal}[^.]*{r_kw}|{r_kw}[^.]*{signal}'
                                if re.search(pattern, content_lower):
                                    key = f"{market} - {region}"
                                    if not any(e['key'] == key for e in expansions):
                                        expansions.append({
                                            'key': key,
                                            'market': market,
                                            'region': region,
                                            'confidence': 0.7
                                        })
                                    break

        return expansions[:15]

    def _extract_financial_highlights(self, content: str) -> Dict:
        """Extract financial highlights and investment focus areas"""
        content_lower = content.lower()

        highlights = {
            'revenue_focus_areas': [],
            'capex_focus_areas': [],
            'investment_priorities': [],
            'cost_initiatives': []
        }

        # Revenue focus patterns
        revenue_patterns = [
            r'revenue\s+(?:from|in|growth)\s+([^.]{10,80})',
            r'(?:strong|significant|growing)\s+(?:revenue|sales)\s+in\s+([^.]{10,60})',
        ]

        # CapEx patterns
        capex_patterns = [
            r'(?:capex|capital\s+expenditure|capital\s+investment)\s+(?:in|for|on)\s+([^.]{10,80})',
            r'(?:invested|investing|investment)\s+(?:in|into)\s+([^.]{10,80})',
        ]

        for pattern in revenue_patterns:
            matches = re.findall(pattern, content_lower)
            for match in matches[:5]:
                if len(match.strip()) > 10:
                    highlights['revenue_focus_areas'].append(match.strip().title())

        for pattern in capex_patterns:
            matches = re.findall(pattern, content_lower)
            for match in matches[:5]:
                if len(match.strip()) > 10:
                    highlights['capex_focus_areas'].append(match.strip().title())

        # Check for specific investment mentions
        investment_areas = [
            'low carbon', 'renewable', 'digital', 'technology', 'efficiency',
            'hydrogen', 'carbon capture', 'electric', 'biofuel'
        ]

        for area in investment_areas:
            if area in content_lower:
                for signal in ['billion', 'million', 'investment', 'capex', 'spending']:
                    if signal in content_lower:
                        # Check proximity
                        idx = content_lower.find(area)
                        context = content_lower[max(0, idx-100):idx+100]
                        if signal in context:
                            highlights['investment_priorities'].append(area.title())
                            break

        return highlights

    def _extract_future_outlook(self, content: str) -> List[str]:
        """Extract future outlook and forward-looking statements"""
        outlook = []
        content_lower = content.lower()

        # Future outlook patterns
        outlook_patterns = [
            r'(?:we\s+expect|we\s+anticipate|we\s+plan)\s+to\s+([^.]{10,100})',
            r'(?:outlook|forecast|projection)\s+(?:for|shows|indicates)\s+([^.]{10,100})',
            r'(?:by\s+2030|by\s+2035|by\s+2040|by\s+2050)[^.]*([^.]{10,100})',
            r'future\s+(?:growth|development|focus)\s+(?:in|on|will)\s+([^.]{10,100})',
            r'(?:target|goal|objective)\s+(?:is|to)\s+([^.]{10,100})',
        ]

        for pattern in outlook_patterns:
            matches = re.findall(pattern, content_lower)
            for match in matches[:5]:
                clean_match = match.strip()
                if len(clean_match) > 15 and len(clean_match) < 150:
                    outlook.append(clean_match.title())

        # Check for specific future targets
        future_targets = [
            'net zero by', 'carbon neutral by', '2030 target', '2040 target', '2050 target',
            'reduce emissions by', 'renewable capacity', 'production target'
        ]

        for target in future_targets:
            if target in content_lower:
                # Get context
                idx = content_lower.find(target)
                context = content_lower[idx:idx+150]
                end_idx = context.find('.')
                if end_idx > 0:
                    outlook.append(context[:end_idx].strip().title())

        return list(set(outlook))[:10]

    def _extract_key_projects(self, content: str) -> List[Dict]:
        """Extract key projects mentioned in the annual report"""
        projects = []
        content_lower = content.lower()

        # Project indicators
        project_patterns = [
            r'(?:project|development|facility)\s+([A-Z][a-zA-Z\s]{3,30})',
            r'([A-Z][a-zA-Z\s]{3,30})\s+(?:project|development|facility)',
        ]

        for pattern in project_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches[:10]:
                clean_match = match.strip().title()
                if len(clean_match) > 4 and len(clean_match) < 50:
                    # Skip common words
                    skip_words = ['the', 'our', 'this', 'that', 'new', 'first', 'second']
                    if clean_match.lower().split()[0] not in skip_words:
                        if not any(p['name'] == clean_match for p in projects):
                            projects.append({
                                'name': clean_match,
                                'status': 'Mentioned',
                                'confidence': 0.6
                            })

        return projects[:15]

    def _extract_partnerships(self, content: str) -> List[Dict]:
        """Extract partnership and collaboration mentions"""
        partnerships = []
        content_lower = content.lower()

        # Partnership patterns
        partnership_patterns = [
            r'partnership\s+with\s+([^,.]{5,50})',
            r'collaboration\s+with\s+([^,.]{5,50})',
            r'joint\s+venture\s+with\s+([^,.]{5,50})',
            r'agreement\s+with\s+([^,.]{5,50})',
            r'working\s+with\s+([^,.]{5,50})',
        ]

        for pattern in partnership_patterns:
            matches = re.findall(pattern, content_lower)
            for match in matches[:10]:
                clean_match = match.strip().title()
                if len(clean_match) > 3 and len(clean_match) < 60:
                    if not any(p['partner'] == clean_match for p in partnerships):
                        partnerships.append({
                            'partner': clean_match,
                            'focus_area': 'General'
                        })

        return partnerships[:10]

    def _extract_risk_factors(self, content: str) -> List[str]:
        """Extract risk factors mentioned in the report"""
        risks = []
        content_lower = content.lower()

        # Risk categories
        risk_keywords = {
            'Climate Risk': ['climate risk', 'climate change risk', 'transition risk'],
            'Regulatory Risk': ['regulatory risk', 'regulatory change', 'compliance risk'],
            'Market Volatility': ['market volatility', 'price volatility', 'commodity price'],
            'Operational Risk': ['operational risk', 'safety risk', 'operational disruption'],
            'Cybersecurity Risk': ['cyber risk', 'cybersecurity', 'data security'],
            'Geopolitical Risk': ['geopolitical', 'political risk', 'sanctions'],
            'Supply Chain Risk': ['supply chain risk', 'supply disruption'],
            'Financial Risk': ['financial risk', 'liquidity risk', 'credit risk'],
        }

        for risk_name, keywords in risk_keywords.items():
            for kw in keywords:
                if kw in content_lower:
                    if risk_name not in risks:
                        risks.append(risk_name)
                    break

        return risks

    def analyze_all_annual_reports(self, annual_reports: List[Dict]) -> Dict[str, AnnualReportAnalysis]:
        """Analyze all annual reports and return dict by company name"""
        results = {}

        for report in annual_reports:
            company = report.get('company', 'Unknown')
            year = report.get('year', 2025)
            content = report.get('content', '')
            url = report.get('url', '')

            if content:
                analysis = self.analyze_annual_report(company, year, content, url)
                results[company] = analysis

        return results

    def save_analysis(self, profiles: List[CompanyProfile], filename: str = 'ai_analysis_results.json'):
        """Save analysis results to JSON file"""
        # Convert profiles to dictionaries for JSON serialization
        results = []
        for profile in profiles:
            profile_dict = {
                'company': profile.company,
                'market_segments': [
                    {
                        'name': seg.name,
                        'confidence': seg.confidence,
                        'evidence': seg.evidence  # Now contains keyword, mention_count, and urls
                    }
                    for seg in profile.market_segments
                ],
                'technologies': [
                    {
                        'name': tech.name,
                        'category': tech.category,
                        'confidence': tech.confidence,
                        'evidence': tech.evidence  # Now contains keyword, mention_count, and urls
                    }
                    for tech in profile.technologies
                ],
                'new_market_opportunities': [
                    {
                        'name': opp.name,
                        'confidence': opp.confidence,
                        'evidence': opp.evidence,  # Now contains keyword, mention_count, and urls
                        'potential_category': opp.potential_category
                    }
                    for opp in profile.new_market_opportunities
                ],
                'sustainability_focus': profile.sustainability_focus,
                'innovation_score': profile.innovation_score,
                'geographic_presence': profile.geographic_presence,
                'annual_report_analysis': self._serialize_annual_report_analysis(profile.annual_report_analysis) if profile.annual_report_analysis else None,
                'summary': profile.summary
            }
            results.append(profile_dict)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"AI analysis results saved to {filename}")
        return results

    def _serialize_annual_report_analysis(self, analysis: AnnualReportAnalysis) -> Dict:
        """Serialize AnnualReportAnalysis to dictionary"""
        return {
            'company': analysis.company,
            'year': analysis.year,
            'strategic_priorities': analysis.strategic_priorities,
            'technology_investments': analysis.technology_investments,
            'market_expansions': analysis.market_expansions,
            'financial_highlights': analysis.financial_highlights,
            'future_outlook': analysis.future_outlook,
            'key_projects': analysis.key_projects,
            'partnerships': analysis.partnerships,
            'risk_factors': analysis.risk_factors,
            'source_url': analysis.source_url
        }