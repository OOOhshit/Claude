#!/usr/bin/env python3
"""
AI-powered analysis module for oil company market presence and technologies.
"""

import json
import re
from typing import List, Dict, Set
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
class CompanyProfile:
    company: str
    market_segments: List[MarketSegment]
    technologies: List[Technology]
    sustainability_focus: float
    innovation_score: float
    geographic_presence: List[str]
    new_market_opportunities: List[NewMarketOpportunity]
    summary: str

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
    
    def extract_market_segments(self, content: str) -> List[MarketSegment]:
        """Extract market segments from content with confidence scores"""
        segments = []
        content_lower = content.lower()
        
        for segment_name, segment_data in self.market_keywords.items():
            evidence = []
            keyword_count = 0
            
            for keyword in segment_data['keywords']:
                matches = len(re.findall(r'\b' + re.escape(keyword) + r'\b', content_lower))
                if matches > 0:
                    keyword_count += matches
                    evidence.append(f"{keyword} (mentioned {matches} times)")
            
            if keyword_count > 0:
                # Calculate confidence based on keyword frequency and weight
                confidence = min(0.95, (keyword_count * segment_data['weight']) / 10)
                
                segments.append(MarketSegment(
                    name=segment_name.replace('_', ' ').title(),
                    confidence=confidence,
                    evidence=evidence[:5]  # Limit evidence items
                ))
        
        return sorted(segments, key=lambda x: x.confidence, reverse=True)
    
    def extract_technologies(self, content: str) -> List[Technology]:
        """Extract technologies from content with categorization"""
        technologies = []
        content_lower = content.lower()
        
        for tech_name, tech_data in self.technology_keywords.items():
            evidence = []
            keyword_count = 0
            
            for keyword in tech_data['keywords']:
                matches = len(re.findall(r'\b' + re.escape(keyword) + r'\b', content_lower))
                if matches > 0:
                    keyword_count += matches
                    evidence.append(f"{keyword} (mentioned {matches} times)")
            
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
    
    def generate_summary(self, profile: CompanyProfile) -> str:
        """Generate a comprehensive summary including new market opportunities"""
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
        
        if len(profile.geographic_presence) > 5:
            summary += f"Global presence with operations in {len(profile.geographic_presence)} regions."
        
        return summary
    
    def detect_new_market_opportunities(self, content: str) -> List[NewMarketOpportunity]:
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
            evidence = []
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
                
                evidence.append(f"Mentioned {market_mentions} times")
                
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
        
        # Combine all content
        all_content = " ".join([item.get('content', '') for item in scraped_data])
        
        # Extract different aspects
        market_segments = self.extract_market_segments(all_content)
        technologies = self.extract_technologies(all_content)
        sustainability_focus = self.calculate_sustainability_focus(all_content)
        innovation_score = self.calculate_innovation_score(all_content)
        geographic_presence = self.extract_geographic_presence(all_content)
        new_market_opportunities = self.detect_new_market_opportunities(all_content)
        
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
                        'evidence': seg.evidence
                    }
                    for seg in profile.market_segments
                ],
                'technologies': [
                    {
                        'name': tech.name,
                        'category': tech.category,
                        'confidence': tech.confidence,
                        'evidence': tech.evidence
                    }
                    for tech in profile.technologies
                ],
                'new_market_opportunities': [
                    {
                        'name': opp.name,
                        'confidence': opp.confidence,
                        'evidence': opp.evidence,
                        'potential_category': opp.potential_category
                    }
                    for opp in profile.new_market_opportunities
                ],
                'sustainability_focus': profile.sustainability_focus,
                'innovation_score': profile.innovation_score,
                'geographic_presence': profile.geographic_presence,
                'summary': profile.summary
            }
            results.append(profile_dict)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"AI analysis results saved to {filename}")
        return results