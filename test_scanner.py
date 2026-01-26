#!/usr/bin/env python3
"""
Test script for the oil company scanner
"""

import json
from ai_analyzer import AIAnalyzer

def test_analyzer():
    """Test the AI analyzer with sample data"""
    print("Testing AI Analyzer...")
    
    # Sample scraped content for testing
    sample_data = [
        {
            'company': 'Test Company',
            'url': 'https://example.com/technology',
            'title': 'Digital Innovation and AI Technologies',
            'content': 'Our company is investing heavily in artificial intelligence and machine learning technologies for upstream exploration and production. We are developing digital twin solutions for offshore drilling operations. Our renewable energy portfolio includes solar and wind projects. We are committed to carbon capture and storage technologies. We are exploring emerging quantum computing market and next-generation space energy solutions.',
            'page_type': 'technology'
        },
        {
            'company': 'Test Company',
            'url': 'https://example.com/markets',
            'title': 'Market Operations',
            'content': 'We operate in upstream exploration and production across North America and Europe. Our downstream operations include refining and petrochemicals. We have natural gas distribution networks and are expanding our renewable energy investments including hydrogen production.',
            'page_type': 'market'
        }
    ]
    
    analyzer = AIAnalyzer()
    profiles = analyzer.analyze_all_companies(sample_data)
    
    if profiles:
        profile = profiles[0]
        print(f"\nAnalysis Results for {profile.company}:")
        print(f"Market Segments Found: {len(profile.market_segments)}")
        for segment in profile.market_segments:
            print(f"  - {segment.name}: {segment.confidence:.1%} confidence")
        
        print(f"\nTechnologies Found: {len(profile.technologies)}")
        for tech in profile.technologies:
            print(f"  - {tech.name} ({tech.category}): {tech.confidence:.1%} confidence")
        
        print(f"\nSustainability Focus: {profile.sustainability_focus:.1%}")
        print(f"Innovation Score: {profile.innovation_score:.1%}")
        print(f"Geographic Presence: {', '.join(profile.geographic_presence)}")
        
        print(f"\nNew Market Opportunities: {len(profile.new_market_opportunities)}")
        for opportunity in profile.new_market_opportunities:
            print(f"  - {opportunity.name} ({opportunity.potential_category}): {opportunity.confidence:.1%} confidence")
        
        print(f"\nSummary: {profile.summary}")
        
        print("\n[OK] AI Analyzer test completed successfully!")
    else:
        print("[ERROR] AI Analyzer test failed!")

def test_company_data():
    """Test loading company data"""
    print("Testing company data loading...")
    
    try:
        with open('companies.json', 'r') as f:
            companies = json.load(f)
        
        print(f"[OK] Loaded {len(companies)} companies:")
        for company in companies:
            print(f"  - {company['name']}: {company['url']}")
        
    except FileNotFoundError:
        print("[ERROR] companies.json not found!")
    except json.JSONDecodeError:
        print("[ERROR] Invalid JSON in companies.json!")

if __name__ == "__main__":
    print("Oil Company Scanner - Test Suite")
    print("=" * 40)
    
    test_company_data()
    print()
    test_analyzer()
    
    print("\n" + "=" * 40)
    print("Test completed. Run 'py main.py' to start the full analysis.")