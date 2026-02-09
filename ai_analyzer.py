#!/usr/bin/env python3
"""
AI-powered analysis module for oil company market presence and technologies.
"""

import json
import re
from typing import List, Dict, Set, Optional
from dataclasses import dataclass
from urllib.parse import urlparse
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
class BusinessActivity:
    """Represents a business activity like partnership, investment, JV, etc."""
    activity_type: str  # 'partnership', 'investment', 'joint_venture', 'acquisition', 'agreement'
    description: str
    partners: List[str]  # Companies/entities involved
    focus_area: str  # What the activity is about
    confidence: float
    evidence: List[Dict]  # Contains keyword, context, urls

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
    business_activities: List[BusinessActivity] = None  # Partnerships, investments, JVs
    annual_report_analysis: Optional[AnnualReportAnalysis] = None
    summary: str = ""

    def __post_init__(self):
        if self.business_activities is None:
            self.business_activities = []

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

        # Base technology keywords (will be enhanced from technology.json)
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

        # Load and integrate technologies from technology.json
        self.technology_keywords.update(self.load_technologies_from_json())

        self.geographic_indicators = [
            'north america', 'usa', 'canada', 'mexico',
            'europe', 'uk', 'norway', 'netherlands', 'france',
            'middle east', 'saudi arabia', 'uae', 'qatar',
            'africa', 'nigeria', 'angola', 'egypt',
            'asia pacific', 'china', 'india', 'australia', 'singapore',
            'south america', 'brazil', 'argentina', 'venezuela'
        ]

        # --- Sentiment / context patterns for contextual analysis ---
        # Negation patterns: when these appear near a keyword, the mention is negative
        self.negation_patterns = [
            r'no\s+plans?\s+(?:for|to)',
            r'not\s+(?:currently\s+)?(?:pursuing|investing|developing|planning|exploring)',
            r'(?:leaving|exiting|divesting|withdrawing|abandoning)\s+(?:the\s+)?',
            r'(?:sold|selling|divested|closed|shutting\s+down|shut\s+down)\s+(?:its\s+|our\s+)?',
            r'(?:ruled\s+out|rejected|cancelled|canceled|scrapped|halted)',
            r'(?:no\s+longer|ceased|stopped|ended|terminated)\s+',
            r'(?:unlikely|improbable|not\s+viable|not\s+feasible)',
            r'(?:reduced|reducing|cutting)\s+(?:its\s+|our\s+)?(?:exposure|presence|investment)\s+in',
        ]

        # Positive engagement patterns: when these appear near a keyword, the mention is positive
        self.positive_patterns = [
            r'(?:entering|expanding|investing|launching|developing|building|constructing)',
            r'(?:partnership|partnered|partnering|collaboration|collaborating)\s+(?:with|in|on|for)',
            r'(?:acquired|acquiring|acquisition|purchased|buying)',
            r'(?:new|major|significant|strategic)\s+(?:investment|project|initiative|venture)',
            r'(?:committed|commitment|pledged|targeting|aiming)\s+(?:to|for)',
            r'(?:growing|growth|increase|increasing|scaling|accelerating)',
            r'(?:leader|leading|pioneer|pioneering|forefront|first-mover)',
            r'(?:commissioned|inaugurated|opened|started|commenced|began)',
            r'(?:breakthrough|advanced|cutting-edge|state-of-the-art|world-class)',
            r'(?:joint\s+venture|jv|alliance|agreement|mou|memorandum)',
        ]

        # Semantic aliases: map alternative terms to canonical keywords
        self.semantic_aliases = {
            'hydrogen': ['blue fuel', 'clean molecules', 'h2', 'green hydrogen', 'blue hydrogen',
                         'hydrogen economy', 'hydrogen hub', 'hydrogen valley'],
            'carbon capture': ['carbon removal', 'co2 capture', 'emission capture',
                               'direct air capture', 'dac', 'carbon sequestration'],
            'renewable': ['clean power', 'green power', 'green energy', 'zero-carbon energy'],
            'solar': ['photovoltaic', 'pv', 'solar farm', 'solar plant'],
            'wind': ['wind farm', 'wind power', 'offshore wind', 'onshore wind'],
            'biofuel': ['sustainable aviation fuel', 'saf', 'renewable diesel',
                        'hvo', 'bio-jet', 'green diesel'],
            'digital twin': ['virtual model', 'virtual replica', 'simulation model'],
            'automation': ['robotic process', 'autonomous operations', 'unmanned operations'],
            'lng': ['liquefied natural gas', 'liquid natural gas'],
            'electric vehicle': ['ev charging', 'e-mobility', 'electrification of transport'],
        }
    
    def load_known_markets(self, filename: str = 'market.json') -> Set[str]:
        """Load known markets from market.json for new market detection"""
        try:
            with open(filename, 'r') as f:
                market_data = json.load(f)

            known_markets = set()

            # Support flat dictionary format: {"Category": ["item1", "item2", ...]}
            if isinstance(market_data, dict) and 'BusinessStructure' not in market_data:
                for category, items in market_data.items():
                    # Add the category name itself
                    clean_category = category.lower().strip()
                    known_markets.add(clean_category)
                    # Add variations of category name
                    known_markets.add(clean_category.replace('/', ' ').replace('-', ' ').replace('&', 'and'))

                    # Add all items in the category
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, str):
                                clean_item = item.lower().strip()
                                known_markets.add(clean_item)
                                # Add variations
                                clean_item_var = clean_item.replace('/', ' ').replace('-', ' ').replace('&', 'and')
                                known_markets.add(clean_item_var)
            else:
                # Legacy BusinessStructure format
                for business_line in market_data.get('BusinessStructure', []):
                    for sub_bl in business_line.get('sub_business_lines', []):
                        for category in sub_bl.get('categories', []):
                            for item in category.get('items', []):
                                clean_item = item.lower().strip()
                                known_markets.add(clean_item)
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

            # Support flat dictionary format: {"Category": ["item1", "item2", ...]}
            if isinstance(market_data, dict) and 'BusinessStructure' not in market_data:
                for category, items in market_data.items():
                    if isinstance(items, list) and items:
                        segment_key = category.lower().replace(' ', '_').replace('&', 'and').replace('/', '_')
                        # Create keywords from items, filtering out empty strings
                        keywords = [item.lower() for item in items if isinstance(item, str) and item.strip()]
                        # Also add the category name as a keyword
                        keywords.append(category.lower())

                        segments[segment_key] = {
                            'keywords': keywords,
                            'weight': 1.5,  # Higher weight for market.json terms
                            'business_line': 'Market',
                            'category': category
                        }
            else:
                # Legacy BusinessStructure format
                for business_line in market_data.get('BusinessStructure', []):
                    bl_name = business_line.get('name', '')

                    for sub_bl in business_line.get('sub_business_lines', []):
                        sub_name = sub_bl.get('name', '')

                        for category in sub_bl.get('categories', []):
                            cat_name = category.get('name', '')

                            if cat_name and category.get('items'):
                                segment_key = cat_name.lower().replace(' ', '_').replace('&', 'and')
                                segments[segment_key] = {
                                    'keywords': [item.lower() for item in category.get('items', [])],
                                    'weight': 1.5,
                                    'business_line': bl_name,
                                    'category': cat_name
                                }

            logger.info(f"Enhanced market detection with {len(segments)} segments from market.json")
            return segments

        except Exception as e:
            logger.error(f"Error extracting market segments from JSON: {str(e)}")
            return {}

    def load_technologies_from_json(self, filename: str = 'technology.json') -> Dict[str, Dict]:
        """Load technologies from technology.json to enhance technology detection"""
        try:
            with open(filename, 'r') as f:
                tech_data = json.load(f)

            tech_keywords = {}

            # Get technology list from the JSON
            technologies = tech_data.get('technology', [])

            if isinstance(technologies, list):
                # Group technologies by category based on naming patterns
                category_patterns = {
                    'Hydrogen': ['hydrogen', 'ammonia cracking', 'electroly', 'fuel cell', 'green ammonia'],
                    'Carbon Capture': ['ccus', 'carbon', 'co2', 'direct air capture', 'cpu'],
                    'Refining': ['refin', 'crude', 'fcc', 'hydrocrack', 'catalytic', 'distillation'],
                    'Petrochemicals': ['ethylene', 'propylene', 'polymer', 'polyethylene', 'polypropylene',
                                       'styrene', 'benzene', 'aromatic', 'pvc', 'pet', 'pta'],
                    'Biofuels': ['bio', 'hefa', 'saf', 'sustainable aviation', 'renewable diesel',
                                 'ethanol', 'biodiesel', 'atj', 'alcohol-to-jet'],
                    'Digital': ['digital', 'iems', 'spyro', 'sam'],
                    'Fertilizers': ['ammonia', 'urea', 'nitric acid', 'fertilizer', 'granulation',
                                    'phosphoric', 'sulfuric'],
                    'Gas Processing': ['lng', 'gas', 'cryomax', 'nitrogen removal', 'ngl'],
                    'Recycling': ['recycl', 'pyrolysis', 'r-pet', 'r-polymer', 'r-pvc', 'plas-tcat', 'volcat'],
                    'Process Equipment': ['burner', 'reformer', 'reactor', 'furnace', 'coil', 'tray',
                                          'heat exchang', 'psa', 'membrane']
                }

                # Categorize each technology
                for tech in technologies:
                    if not isinstance(tech, str) or not tech.strip():
                        continue

                    tech_lower = tech.lower()
                    assigned_category = 'General Technology'

                    # Find matching category
                    for category, patterns in category_patterns.items():
                        if any(pattern in tech_lower for pattern in patterns):
                            assigned_category = category
                            break

                    # Create a unique key for this technology
                    tech_key = tech_lower.replace(' ', '_').replace('/', '_').replace('-', '_')[:50]

                    # Add as individual technology entry for precise matching
                    tech_keywords[f'tech_{tech_key}'] = {
                        'keywords': [tech_lower, tech_lower.replace('/', ' '), tech_lower.replace('-', ' ')],
                        'category': assigned_category
                    }

                logger.info(f"Loaded {len(tech_keywords)} technologies from technology.json")

            return tech_keywords

        except FileNotFoundError:
            logger.warning("technology.json not found, using default technology keywords")
            return {}
        except Exception as e:
            logger.error(f"Error loading technologies from JSON: {str(e)}")
            return {}

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences for contextual analysis."""
        # Split on sentence-ending punctuation, keeping reasonable chunks
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Also split very long segments on semicolons/colons
        result = []
        for s in sentences:
            if len(s) > 500:
                result.extend(re.split(r'[;:]\s+', s))
            else:
                result.append(s)
        return [s.strip() for s in result if len(s.strip()) > 10]

    def _analyze_sentence_sentiment(self, sentence: str, keyword: str) -> str:
        """Determine whether a keyword mention in a sentence is positive, negative, or neutral.

        Returns: 'positive', 'negative', or 'neutral'
        """
        sentence_lower = sentence.lower()

        # Check for negation patterns near the keyword
        for neg_pattern in self.negation_patterns:
            # Look for negation within 60 chars before the keyword
            kw_idx = sentence_lower.find(keyword.lower())
            if kw_idx == -1:
                continue
            window_before = sentence_lower[max(0, kw_idx - 80):kw_idx]
            window_after = sentence_lower[kw_idx:kw_idx + len(keyword) + 80]
            context_window = window_before + ' ' + window_after
            if re.search(neg_pattern, context_window):
                return 'negative'

        # Check for positive engagement patterns
        for pos_pattern in self.positive_patterns:
            kw_idx = sentence_lower.find(keyword.lower())
            if kw_idx == -1:
                continue
            window_before = sentence_lower[max(0, kw_idx - 80):kw_idx]
            window_after = sentence_lower[kw_idx:kw_idx + len(keyword) + 80]
            context_window = window_before + ' ' + window_after
            if re.search(pos_pattern, context_window):
                return 'positive'

        return 'neutral'

    def _is_boilerplate_context(self, sentence: str) -> bool:
        """Detect if a sentence is likely boilerplate (footer, nav, cookie banner, etc.)."""
        boilerplate_signals = [
            'cookie', 'privacy policy', 'terms of use', 'terms and conditions',
            'all rights reserved', 'copyright', 'subscribe to', 'sign up for',
            'follow us', 'contact us', 'click here', 'learn more',
            'accept all', 'manage preferences', 'cookie settings',
            'site map', 'back to top', 'skip to content',
        ]
        sentence_lower = sentence.lower()
        return any(signal in sentence_lower for signal in boilerplate_signals)

    def _get_page_type_weight(self, url: str, keyword: str) -> float:
        """Give higher weight to keywords found on dedicated/relevant pages.

        A mention of 'hydrogen' on a /hydrogen page is worth more than
        a mention in a generic /about page.
        """
        if not url:
            return 1.0

        url_lower = url.lower()
        keyword_lower = keyword.lower()

        # Strong signal: keyword appears in the URL path itself
        url_path = urlparse(url_lower).path.replace('-', ' ').replace('/', ' ').replace('_', ' ')
        if keyword_lower in url_path:
            return 2.0

        # Medium signal: URL is a topic-related section
        topic_url_patterns = {
            'technology': 1.5, 'innovation': 1.5, 'research': 1.5,
            'strategy': 1.4, 'business': 1.3, 'operations': 1.3,
            'sustainability': 1.4, 'energy': 1.2, 'investor': 1.3,
            'news': 1.1, 'press': 1.1, 'project': 1.3,
        }
        for pattern, weight in topic_url_patterns.items():
            if pattern in url_path:
                return weight

        return 1.0

    def _find_keyword_with_aliases(self, keyword: str, text: str) -> List[str]:
        """Find a keyword or any of its semantic aliases in text.

        Returns list of actual matched terms.
        """
        text_lower = text.lower()
        found = []

        if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', text_lower):
            found.append(keyword)

        # Check semantic aliases
        for canonical, aliases in self.semantic_aliases.items():
            if keyword.lower() == canonical or keyword.lower() in aliases:
                terms_to_check = [canonical] + aliases
                for alias in terms_to_check:
                    if alias != keyword.lower() and re.search(r'\b' + re.escape(alias) + r'\b', text_lower):
                        found.append(alias)

        return found

    def extract_market_segments(self, content: str, url_content_map: Dict[str, str] = None) -> List[MarketSegment]:
        """Extract market segments using contextual sentence-level analysis.

        Instead of counting keyword frequency, this method:
          1. Splits content into sentences
          2. For each keyword match, checks the sentence context for negation
             or positive signals
          3. Weights mentions by page type (dedicated page vs generic)
          4. Filters out boilerplate mentions (cookie banners, footers)
          5. Uses semantic aliases for broader matching
        """
        segments = []
        sentences = self._split_sentences(content)

        for segment_name, segment_data in self.market_keywords.items():
            positive_score = 0.0
            negative_count = 0
            evidence = []
            seen_evidence_keys = set()

            for keyword in segment_data['keywords']:
                for sentence in sentences:
                    # Skip boilerplate
                    if self._is_boilerplate_context(sentence):
                        continue

                    # Check keyword + aliases
                    matched_terms = self._find_keyword_with_aliases(keyword, sentence)
                    if not matched_terms:
                        continue

                    # Analyze sentiment of this mention
                    sentiment = self._analyze_sentence_sentiment(sentence, matched_terms[0])
                    if sentiment == 'negative':
                        negative_count += 1
                        continue

                    # Score: positive mentions get more weight
                    mention_score = 1.5 if sentiment == 'positive' else 0.8

                    # Apply page-type weighting via url_content_map
                    if url_content_map:
                        for url, page_content in url_content_map.items():
                            if any(re.search(r'\b' + re.escape(t) + r'\b', page_content.lower()) for t in matched_terms):
                                page_weight = self._get_page_type_weight(url, keyword)
                                mention_score *= page_weight

                                ev_key = f"{keyword}:{url}"
                                if ev_key not in seen_evidence_keys and len(evidence) < 5:
                                    seen_evidence_keys.add(ev_key)
                                    evidence.append({
                                        'keyword': matched_terms[0],
                                        'context': sentence[:200],
                                        'sentiment': sentiment,
                                        'urls': [url]
                                    })
                                break
                    else:
                        ev_key = f"{keyword}:{sentence[:50]}"
                        if ev_key not in seen_evidence_keys and len(evidence) < 5:
                            seen_evidence_keys.add(ev_key)
                            evidence.append({
                                'keyword': matched_terms[0],
                                'context': sentence[:200],
                                'sentiment': sentiment,
                                'urls': []
                            })

                    positive_score += mention_score

            # Only include segments with net positive evidence
            if positive_score > 0 and positive_score > negative_count:
                # Confidence based on weighted score, not raw count
                weight = segment_data.get('weight', 1.0)
                confidence = min(0.95, (positive_score * weight) / 12)

                # Penalize confidence if there are significant negative mentions
                if negative_count > 0:
                    neg_ratio = negative_count / (positive_score + negative_count)
                    confidence *= (1.0 - neg_ratio * 0.5)

                segments.append(MarketSegment(
                    name=segment_name.replace('_', ' ').title(),
                    confidence=round(confidence, 3),
                    evidence=evidence[:5]
                ))

        return sorted(segments, key=lambda x: x.confidence, reverse=True)
    
    def extract_technologies(self, content: str, url_content_map: Dict[str, str] = None) -> List[Technology]:
        """Extract technologies using contextual sentence-level analysis.

        Same approach as extract_market_segments: analyses sentence context,
        handles negation, weights by page type, and uses semantic aliases.
        """
        technologies = []
        sentences = self._split_sentences(content)

        for tech_name, tech_data in self.technology_keywords.items():
            positive_score = 0.0
            negative_count = 0
            evidence = []
            seen_evidence_keys = set()

            for keyword in tech_data['keywords']:
                for sentence in sentences:
                    if self._is_boilerplate_context(sentence):
                        continue

                    matched_terms = self._find_keyword_with_aliases(keyword, sentence)
                    if not matched_terms:
                        continue

                    sentiment = self._analyze_sentence_sentiment(sentence, matched_terms[0])
                    if sentiment == 'negative':
                        negative_count += 1
                        continue

                    mention_score = 1.5 if sentiment == 'positive' else 0.8

                    if url_content_map:
                        for url, page_content in url_content_map.items():
                            if any(re.search(r'\b' + re.escape(t) + r'\b', page_content.lower()) for t in matched_terms):
                                page_weight = self._get_page_type_weight(url, keyword)
                                mention_score *= page_weight

                                ev_key = f"{keyword}:{url}"
                                if ev_key not in seen_evidence_keys and len(evidence) < 3:
                                    seen_evidence_keys.add(ev_key)
                                    evidence.append({
                                        'keyword': matched_terms[0],
                                        'context': sentence[:200],
                                        'sentiment': sentiment,
                                        'urls': [url]
                                    })
                                break
                    else:
                        ev_key = f"{keyword}:{sentence[:50]}"
                        if ev_key not in seen_evidence_keys and len(evidence) < 3:
                            seen_evidence_keys.add(ev_key)
                            evidence.append({
                                'keyword': matched_terms[0],
                                'context': sentence[:200],
                                'sentiment': sentiment,
                                'urls': []
                            })

                    positive_score += mention_score

            if positive_score > 0 and positive_score > negative_count:
                confidence = min(0.95, positive_score / 6)
                if negative_count > 0:
                    neg_ratio = negative_count / (positive_score + negative_count)
                    confidence *= (1.0 - neg_ratio * 0.5)

                technologies.append(Technology(
                    name=tech_name.replace('_', ' ').title(),
                    category=tech_data['category'],
                    confidence=round(confidence, 3),
                    evidence=evidence[:3]
                ))

        return sorted(technologies, key=lambda x: x.confidence, reverse=True)
    
    def calculate_sustainability_focus(self, content: str) -> float:
        """Calculate sustainability focus score using contextual analysis.

        Only counts mentions in substantive sentences (not boilerplate),
        and weights positive commitments higher than neutral mentions.
        """
        sustainability_keywords = [
            'sustainability', 'sustainable', 'renewable', 'clean energy',
            'carbon neutral', 'net zero', 'emissions reduction',
            'environmental', 'green energy', 'climate change',
            'energy transition', 'decarbonization', 'decarbonisation',
        ]

        sentences = self._split_sentences(content)
        weighted_score = 0.0

        for sentence in sentences:
            if self._is_boilerplate_context(sentence):
                continue
            sentence_lower = sentence.lower()
            for keyword in sustainability_keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', sentence_lower):
                    sentiment = self._analyze_sentence_sentiment(sentence, keyword)
                    if sentiment == 'negative':
                        continue
                    weighted_score += 1.5 if sentiment == 'positive' else 0.7
                    break  # one keyword per sentence is enough

        return round(min(1.0, weighted_score / 15), 3)

    def calculate_innovation_score(self, content: str) -> float:
        """Calculate innovation score using contextual analysis.

        Weights active innovation signals (investing, launching, pioneering)
        higher than passive mentions.
        """
        innovation_keywords = [
            'innovation', 'research', 'development', 'r&d',
            'breakthrough', 'cutting-edge', 'advanced',
            'pioneering', 'patent', 'technology development',
            'prototype', 'pilot project', 'first-of-its-kind',
        ]

        sentences = self._split_sentences(content)
        weighted_score = 0.0

        for sentence in sentences:
            if self._is_boilerplate_context(sentence):
                continue
            sentence_lower = sentence.lower()
            for keyword in innovation_keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', sentence_lower):
                    sentiment = self._analyze_sentence_sentiment(sentence, keyword)
                    if sentiment == 'negative':
                        continue
                    weighted_score += 1.5 if sentiment == 'positive' else 0.7
                    break

        return round(min(1.0, weighted_score / 12), 3)
    
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
        """Generate a comprehensive summary including new market opportunities, business activities, and annual report insights"""
        top_markets = [seg.name for seg in profile.market_segments[:3]]
        top_techs = [tech.name for tech in profile.technologies[:3]]

        summary = f"{profile.company} operates primarily in {', '.join(top_markets)} markets. "

        if top_techs:
            summary += f"Key technology focus areas include {', '.join(top_techs)}. "

        if profile.sustainability_focus > 0.5:
            summary += "The company shows strong commitment to sustainability and clean energy transition. "

        if profile.innovation_score > 0.5:
            summary += "High innovation activity with significant R&D investments. "

        # Add business activities summary
        if profile.business_activities:
            activity_types = {}
            for activity in profile.business_activities:
                activity_types[activity.activity_type] = activity_types.get(activity.activity_type, 0) + 1

            activity_summary = ", ".join([f"{count} {atype}{'s' if count > 1 else ''}"
                                          for atype, count in activity_types.items()])
            summary += f"[BUSINESS ACTIVITIES] Detected: {activity_summary}. "

            # Highlight key partnerships/JVs
            key_activities = [a for a in profile.business_activities if a.activity_type in ['joint_venture', 'acquisition', 'partnership']][:2]
            if key_activities:
                partners = [a.partners[0] if a.partners else 'Unknown' for a in key_activities]
                summary += f"Key partners include: {', '.join(partners)}. "

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
        """Detect potentially new energy/oil-domain markets not in our known market list.

        Improved approach:
          1. Only matches multi-word phrases that include a domain-specific qualifier
          2. Requires the phrase to appear in a substantive sentence (not boilerplate)
          3. Requires positive/neutral context (not negated)
          4. Filters out generic phrases ('its business', 'the energy', 'of energy')
          5. Validates that the candidate is a real market concept related to oil & energy
        """
        sentences = self._split_sentences(content)

        # Patterns that capture specific market/technology concepts
        # Each pattern requires a domain-qualifying adjective or noun
        new_market_patterns = [
            # "<specific_adjective> <domain_noun>" patterns
            r'\b((?:green|blue|clean|renewable|sustainable|advanced|smart|digital|offshore|onshore|floating|modular)\s+(?:hydrogen|ammonia|methanol|fuel|energy|power|gas|chemical|material|technology|solution|platform))\b',
            # "emerging <specific_domain_term>"
            r'\b(emerging\s+(?:hydrogen|ammonia|carbon|biofuel|lng|lithium|battery|geothermal|nuclear|wind|solar|fuel\s+cell|electrolysis)\s*(?:market|sector|industry|opportunity|technology)?)\b',
            # "next-generation <specific_tech>"
            r'\b(next[\s-]generation\s+(?:biofuel|catalyst|refining|reactor|battery|fuel\s+cell|electrolyzer|turbine|solar|polymer))\b',
            # Specific compound market terms
            r'\b((?:carbon\s+credit|emissions?\s+trading|green\s+bond|sustainable\s+finance|circular\s+economy|waste[\s-]to[\s-]energy|power[\s-]to[\s-]x|e[\s-]fuel|e[\s-]methanol|direct\s+air\s+capture|ocean\s+energy|tidal\s+energy|geothermal\s+energy|small\s+modular\s+reactor|floating\s+wind|green\s+ammonia|blue\s+ammonia))\b',
        ]

        # Words/phrases that should NEVER be considered a new market
        # (too generic or not meaningful in energy context)
        garbage_filters = {
            'its business', 'the energy', 'of energy', 'our business',
            'the market', 'its market', 'the sector', 'the industry',
            'our energy', 'this market', 'this business', 'this sector',
            'global energy', 'world energy', 'new energy', 'total energy',
            'the technology', 'of technology', 'its technology',
            'all energy', 'more energy', 'other energy', 'any energy',
            'clean technology', 'energy technology',  # too broad
            'future energy', 'future market', 'future technology',
            'digital technology', 'smart technology',  # too generic
            'emerging market', 'emerging technology',  # the category itself
        }

        # Domain-relevance check: must contain at least one domain term
        domain_terms = {
            'hydrogen', 'ammonia', 'methanol', 'carbon', 'biofuel', 'lng',
            'lithium', 'battery', 'geothermal', 'nuclear', 'wind', 'solar',
            'fuel cell', 'electrolysis', 'electrolyzer', 'refining', 'catalyst',
            'polymer', 'reactor', 'turbine', 'biomass', 'biogas', 'ethanol',
            'methane', 'syngas', 'tidal', 'ocean', 'floating', 'modular',
            'e-fuel', 'e-methanol', 'circular', 'waste-to-energy', 'power-to-x',
            'co2', 'emissions', 'credit', 'trading', 'capture', 'storage',
            'green bond', 'sustainable finance',
        }

        potential_new_markets = {}  # market_name -> (sentences, urls)

        for sentence in sentences:
            if self._is_boilerplate_context(sentence):
                continue

            sentence_lower = sentence.lower()
            for pattern in new_market_patterns:
                matches = re.findall(pattern, sentence_lower)
                for match in matches:
                    clean_match = match.strip()

                    # Filter: reject garbage / too-generic phrases
                    if clean_match in garbage_filters:
                        continue
                    if len(clean_match) < 8 or len(clean_match) > 60:
                        continue

                    # Filter: must contain a domain-specific term
                    if not any(dt in clean_match for dt in domain_terms):
                        continue

                    # Filter: not in our known markets already
                    if any(known in clean_match or clean_match in known for known in self.known_markets):
                        continue

                    # Check sentiment: skip negated mentions
                    sentiment = self._analyze_sentence_sentiment(sentence, clean_match)
                    if sentiment == 'negative':
                        continue

                    if clean_match not in potential_new_markets:
                        potential_new_markets[clean_match] = {
                            'sentences': [], 'urls': set(), 'positive_count': 0, 'neutral_count': 0
                        }

                    potential_new_markets[clean_match]['sentences'].append(sentence[:200])
                    if sentiment == 'positive':
                        potential_new_markets[clean_match]['positive_count'] += 1
                    else:
                        potential_new_markets[clean_match]['neutral_count'] += 1

                    # Find source URLs
                    if url_content_map:
                        for url, page_content in url_content_map.items():
                            if clean_match in page_content.lower():
                                potential_new_markets[clean_match]['urls'].add(url)

        # Convert to NewMarketOpportunity objects
        new_opportunities = []
        for market, data in potential_new_markets.items():
            total_mentions = data['positive_count'] + data['neutral_count']
            if total_mentions < 1:
                continue

            # Confidence: positive mentions count more; require at least 2 mentions for > 50% confidence
            confidence = min(0.8, (data['positive_count'] * 1.5 + data['neutral_count'] * 0.5) / 5)

            # Categorize the new market
            potential_category = "Emerging Market"
            if any(term in market for term in ['energy', 'fuel', 'power', 'wind', 'solar', 'tidal', 'geothermal']):
                potential_category = "New Energy Solutions"
            elif any(term in market for term in ['hydrogen', 'ammonia', 'methanol', 'e-fuel', 'biofuel']):
                potential_category = "Alternative Fuels & Feedstocks"
            elif any(term in market for term in ['carbon', 'capture', 'co2', 'emission', 'credit']):
                potential_category = "Carbon & Emissions Management"
            elif any(term in market for term in ['battery', 'lithium', 'reactor', 'modular']):
                potential_category = "Technology Innovation"
            elif any(term in market for term in ['chemical', 'material', 'polymer', 'catalyst']):
                potential_category = "New Materials/Chemicals"
            elif any(term in market for term in ['circular', 'waste', 'recycl']):
                potential_category = "Circular Economy"
            elif any(term in market for term in ['bond', 'finance', 'trading']):
                potential_category = "Green Finance"

            evidence = [{
                'keyword': market,
                'context': data['sentences'][0] if data['sentences'] else '',
                'sentiment': 'positive' if data['positive_count'] > 0 else 'neutral',
                'urls': list(data['urls'])[:5]
            }]

            new_opportunities.append(NewMarketOpportunity(
                name=market.title(),
                confidence=round(confidence, 3),
                evidence=evidence,
                potential_category=potential_category
            ))

        return sorted(new_opportunities, key=lambda x: x.confidence, reverse=True)[:5]

    def extract_business_activities(self, content: str, url_content_map: Dict[str, str] = None) -> List[BusinessActivity]:
        """Extract business activities: partnerships, investments, JVs, acquisitions from website content"""
        activities = []
        content_lower = content.lower()

        # Activity patterns with their types
        activity_patterns = {
            'partnership': [
                r'partnership\s+(?:with|between)\s+([^,.]{5,80})',
                r'partnered\s+with\s+([^,.]{5,80})',
                r'strategic\s+partner(?:ship)?\s+(?:with)?\s*([^,.]{5,80})',
                r'partnering\s+with\s+([^,.]{5,80})',
                r'collaboration\s+(?:with|between)\s+([^,.]{5,80})',
                r'collaborating\s+with\s+([^,.]{5,80})',
            ],
            'joint_venture': [
                r'joint\s+venture\s+(?:with|between)\s+([^,.]{5,80})',
                r'jv\s+(?:with|between)\s+([^,.]{5,80})',
                r'joint\s+venture\s+(?:called|named)?\s*([^,.]{5,80})',
                r'formed\s+(?:a\s+)?joint\s+venture\s+([^,.]{5,80})',
            ],
            'investment': [
                r'invest(?:ed|ing|ment)?\s+(?:in|into)\s+([^,.]{5,80})',
                r'acquired?\s+(?:a\s+)?(?:stake|interest|share)\s+in\s+([^,.]{5,80})',
                r'funding\s+(?:for|of|to)\s+([^,.]{5,80})',
                r'backed\s+(?:by)?\s*([^,.]{5,80})',
                r'equity\s+(?:investment|stake)\s+in\s+([^,.]{5,80})',
            ],
            'acquisition': [
                r'acqui(?:red|sition)\s+(?:of)?\s*([^,.]{5,80})',
                r'purchased\s+([^,.]{5,80})',
                r'bought\s+([^,.]{5,80})',
                r'takeover\s+of\s+([^,.]{5,80})',
                r'merger\s+with\s+([^,.]{5,80})',
            ],
            'agreement': [
                r'(?:signed|entered)\s+(?:a\s+)?(?:an\s+)?agreement\s+(?:with)?\s*([^,.]{5,80})',
                r'mou\s+(?:with|between)\s+([^,.]{5,80})',
                r'memorandum\s+of\s+understanding\s+(?:with)?\s*([^,.]{5,80})',
                r'licensing\s+agreement\s+(?:with)?\s*([^,.]{5,80})',
                r'supply\s+agreement\s+(?:with)?\s*([^,.]{5,80})',
                r'offtake\s+agreement\s+(?:with)?\s*([^,.]{5,80})',
            ],
            'alliance': [
                r'(?:strategic\s+)?alliance\s+(?:with|between)\s+([^,.]{5,80})',
                r'allied\s+with\s+([^,.]{5,80})',
                r'consortium\s+(?:with|including)\s+([^,.]{5,80})',
            ]
        }

        # Focus area keywords to categorize activities
        focus_keywords = {
            'Hydrogen': ['hydrogen', 'h2', 'electrolyzer', 'fuel cell', 'green hydrogen', 'blue hydrogen'],
            'Carbon Capture': ['carbon capture', 'ccus', 'ccs', 'co2', 'carbon storage', 'decarbonization'],
            'Renewable Energy': ['solar', 'wind', 'renewable', 'clean energy', 'battery', 'energy storage'],
            'LNG/Gas': ['lng', 'natural gas', 'gas processing', 'liquefaction'],
            'Refining': ['refinery', 'refining', 'fuel', 'petrochemical'],
            'Digital/Technology': ['digital', 'technology', 'ai', 'automation', 'software'],
            'Sustainability': ['sustainability', 'esg', 'net zero', 'emission', 'climate'],
            'Upstream': ['exploration', 'drilling', 'production', 'offshore', 'upstream'],
            'Chemicals': ['chemical', 'polymer', 'plastic', 'petrochemical'],
            'Biofuels': ['biofuel', 'biodiesel', 'saf', 'sustainable aviation', 'renewable diesel'],
        }

        found_activities = {}  # Use dict to avoid duplicates

        for activity_type, patterns in activity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, content_lower, re.IGNORECASE)
                for match in matches:
                    partner_text = match.group(1).strip() if match.groups() else ""

                    # Skip if too short or contains common false positives
                    if len(partner_text) < 5:
                        continue
                    skip_words = ['the', 'a', 'an', 'our', 'their', 'its', 'this', 'that', 'which', 'where']
                    if partner_text.split()[0].lower() in skip_words:
                        partner_text = ' '.join(partner_text.split()[1:])

                    if len(partner_text) < 3:
                        continue

                    # Get context around match for focus area detection
                    start_idx = max(0, match.start() - 150)
                    end_idx = min(len(content_lower), match.end() + 150)
                    context = content_lower[start_idx:end_idx]

                    # Determine focus area
                    focus_area = 'General'
                    for area, keywords in focus_keywords.items():
                        if any(kw in context for kw in keywords):
                            focus_area = area
                            break

                    # Create unique key to avoid duplicates
                    activity_key = f"{activity_type}:{partner_text[:30].lower()}"

                    if activity_key not in found_activities:
                        # Find URLs where this activity is mentioned
                        urls_with_activity = []
                        if url_content_map:
                            for url, page_content in url_content_map.items():
                                if partner_text.lower() in page_content.lower():
                                    urls_with_activity.append(url)

                        found_activities[activity_key] = BusinessActivity(
                            activity_type=activity_type,
                            description=match.group(0).strip()[:200],
                            partners=[partner_text.title()],
                            focus_area=focus_area,
                            confidence=0.7,
                            evidence=[{
                                'keyword': match.group(0)[:100],
                                'context': context[:200],
                                'urls': urls_with_activity[:5]
                            }]
                        )

        # Convert to list and sort by activity type
        activities = list(found_activities.values())

        # Sort by activity type priority
        type_priority = {'acquisition': 0, 'joint_venture': 1, 'investment': 2, 'partnership': 3, 'alliance': 4, 'agreement': 5}
        activities.sort(key=lambda x: type_priority.get(x.activity_type, 10))

        logger.info(f"Extracted {len(activities)} business activities")
        return activities[:20]  # Limit to top 20

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
        business_activities = self.extract_business_activities(all_content, url_content_map)

        # Create profile
        profile = CompanyProfile(
            company=company_name,
            market_segments=market_segments,
            technologies=technologies,
            sustainability_focus=sustainability_focus,
            innovation_score=innovation_score,
            geographic_presence=geographic_presence,
            new_market_opportunities=new_market_opportunities,
            business_activities=business_activities,
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
                'business_activities': [
                    {
                        'activity_type': activity.activity_type,
                        'description': activity.description,
                        'partners': activity.partners,
                        'focus_area': activity.focus_area,
                        'confidence': activity.confidence,
                        'evidence': activity.evidence
                    }
                    for activity in (profile.business_activities or [])
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