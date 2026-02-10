#!/usr/bin/env python3
"""
Test script for the oil company scanner (market-focused version).

Tests market analysis functionality and contextual improvements:
  - Market segment detection from oil & energy domain
  - Negation handling (negative mentions should not count as positive evidence)
  - Context-aware scoring (dedicated pages weighted higher)
  - Boilerplate filtering (cookie banners, footers ignored)
  - New market detection filters out generic phrases
  - Semantic aliases expand keyword matching
  - No technology/AI/IT/digital search
"""

import json
from ai_analyzer import AIAnalyzer


def test_analyzer():
    """Test the AI analyzer with sample data (market-focused)"""
    print("Testing AI Analyzer (market-focused)...")

    sample_data = [
        {
            'company': 'Test Company',
            'url': 'https://example.com/operations',
            'title': 'Our Operations and Market Presence',
            'content': 'Our company is investing heavily in upstream exploration and production. We are expanding offshore drilling operations in deepwater fields. Our renewable energy portfolio includes solar and wind projects. We are committed to carbon capture and storage. We are exploring emerging green ammonia market and floating wind opportunities.',
            'page_type': 'operations'
        },
        {
            'company': 'Test Company',
            'url': 'https://example.com/markets',
            'title': 'Market Operations',
            'content': 'We operate in upstream exploration and production across North America and Europe. Our downstream operations include refining and petrochemicals. We have natural gas distribution networks and are expanding our renewable energy investments including hydrogen production. Our LNG business continues to grow with new projects in Qatar and Australia.',
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

        print(f"\nSustainability Focus: {profile.sustainability_focus:.1%}")
        print(f"Geographic Presence: {', '.join(profile.geographic_presence)}")

        print(f"\nNew Market Opportunities: {len(profile.new_market_opportunities)}")
        for opportunity in profile.new_market_opportunities:
            print(f"  - {opportunity.name} ({opportunity.potential_category}): {opportunity.confidence:.1%} confidence")

        print(f"\nSummary: {profile.summary}")

        print("\n[OK] AI Analyzer market-focused test completed successfully!")
    else:
        print("[ERROR] AI Analyzer test failed!")


def test_negation_handling():
    """Test that negative mentions are correctly handled"""
    print("\nTesting negation handling...")

    analyzer = AIAnalyzer()

    # Content with explicit negation of hydrogen and carbon capture
    negative_data = [
        {
            'company': 'NegTest Corp',
            'url': 'https://example.com/strategy',
            'title': 'Strategy Update',
            'content': (
                'We have no plans for carbon capture investments at this time. '
                'The company is leaving the hydrogen market due to cost concerns. '
                'We are not pursuing wind energy projects in the current portfolio. '
                'We divested from our solar business last year. '
                'We ruled out any involvement in biofuel production.'
            ),
            'page_type': 'market'
        }
    ]

    profiles = analyzer.analyze_all_companies(negative_data)
    if profiles:
        profile = profiles[0]
        segment_names = [s.name.lower() for s in profile.market_segments]

        # These segments should NOT appear (or have very low confidence) because
        # all mentions are negated
        print(f"  Market segments found: {segment_names}")
        issues = []
        for seg in profile.market_segments:
            if seg.confidence > 0.5:
                issues.append(f"    WARNING: '{seg.name}' has high confidence ({seg.confidence:.1%}) despite negation")

        if issues:
            for issue in issues:
                print(issue)
            print("  [WARN] Some negated segments still scored high - review negation patterns")
        else:
            print("  [OK] Negated segments correctly handled (low or no confidence)")
    else:
        print("  [OK] No profiles generated for all-negative content")


def test_boilerplate_filtering():
    """Test that boilerplate content (cookie banners, footers) is ignored"""
    print("\nTesting boilerplate filtering...")

    analyzer = AIAnalyzer()

    # Content where keywords ONLY appear in boilerplate context
    boilerplate_data = [
        {
            'company': 'Boilerplate Corp',
            'url': 'https://example.com/about',
            'title': 'About Us',
            'content': (
                'We use cookies to improve your experience. Accept all cookies. '
                'Privacy policy | Terms of use | Cookie settings. '
                'Subscribe to our newsletter for solar energy updates. '
                'Follow us on social media for hydrogen news. '
                'Click here to learn more about carbon capture. '
                'Back to top. All rights reserved 2025.'
            ),
            'page_type': 'general'
        }
    ]

    profiles = analyzer.analyze_all_companies(boilerplate_data)
    if profiles:
        profile = profiles[0]
        total_segments = len(profile.market_segments)
        print(f"  Market segments from boilerplate content: {total_segments}")
        if total_segments == 0:
            print("  [OK] Boilerplate content correctly filtered out")
        else:
            print(f"  [WARN] {total_segments} segments detected from boilerplate (should be 0)")
            for seg in profile.market_segments:
                print(f"    - {seg.name}: {seg.confidence:.1%}")
    else:
        print("  [OK] No profiles from boilerplate-only content")


def test_new_market_filters_garbage():
    """Test that generic phrases like 'the energy', 'its business' are filtered out"""
    print("\nTesting new market opportunity garbage filtering...")

    analyzer = AIAnalyzer()

    # Content with generic phrases that should NOT be detected as new markets
    generic_data = [
        {
            'company': 'Generic Corp',
            'url': 'https://example.com/about',
            'title': 'About the Company',
            'content': (
                'Its business spans multiple sectors. The energy industry is evolving. '
                'Of energy production, we handle the full lifecycle. '
                'The market is competitive. Our energy solutions are diverse. '
                'The technology landscape changes rapidly. '
                'But we are also investing in green ammonia production and floating wind technology. '
                'Our new direct air capture facility will be operational next year.'
            ),
            'page_type': 'general'
        }
    ]

    profiles = analyzer.analyze_all_companies(generic_data)
    if profiles:
        profile = profiles[0]
        opp_names = [o.name.lower() for o in profile.new_market_opportunities]
        print(f"  New market opportunities detected: {opp_names}")

        garbage_found = [name for name in opp_names
                         if name in ['its business', 'the energy', 'of energy',
                                     'the market', 'the technology']]
        if garbage_found:
            print(f"  [FAIL] Generic phrases detected as markets: {garbage_found}")
        else:
            print("  [OK] No generic garbage phrases detected")

        # Check that real opportunities are still detected
        real_opportunities = [name for name in opp_names
                              if any(term in name for term in ['ammonia', 'wind', 'air capture'])]
        if real_opportunities:
            print(f"  [OK] Real opportunities still detected: {real_opportunities}")
    else:
        print("  [INFO] No profiles generated")


def test_semantic_aliases():
    """Test that semantic aliases expand keyword matching (market-focused)"""
    print("\nTesting semantic alias matching (oil & energy markets)...")

    analyzer = AIAnalyzer()

    # Content using alternative terms (aliases) for oil & energy market keywords
    alias_data = [
        {
            'company': 'Alias Corp',
            'url': 'https://example.com/hydrogen-strategy',
            'title': 'Clean Fuel Strategy',
            'content': (
                'We are expanding our blue fuel production capabilities. '
                'Our clean molecules initiative is a key strategic priority. '
                'The company launched a new photovoltaic farm in Texas. '
                'Sustainable aviation fuel production is accelerating at our facility. '
                'Our e-mobility charging network now covers 500 locations. '
                'We are investing in liquefied natural gas terminals across Asia.'
            ),
            'page_type': 'market'
        }
    ]

    profiles = analyzer.analyze_all_companies(alias_data)
    if profiles:
        profile = profiles[0]
        print(f"  Market segments: {[s.name for s in profile.market_segments]}")

        # Check if alias terms were matched via semantic expansion
        all_evidence_keywords = set()
        for seg in profile.market_segments:
            for ev in seg.evidence:
                if isinstance(ev, dict):
                    all_evidence_keywords.add(ev.get('keyword', '').lower())

        print(f"  Evidence keywords found: {all_evidence_keywords}")
        if any(term in all_evidence_keywords for term in ['blue fuel', 'clean molecules', 'photovoltaic',
                                                           'sustainable aviation fuel', 'e-mobility',
                                                           'liquefied natural gas']):
            print("  [OK] Semantic aliases correctly expanded keyword matching")
        else:
            print("  [INFO] Alias matching may need tuning - check evidence keywords above")
    else:
        print("  [ERROR] No profiles generated for alias test")


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
    print("=" * 60)

    test_company_data()
    print()
    test_analyzer()
    test_negation_handling()
    test_boilerplate_filtering()
    test_new_market_filters_garbage()
    test_semantic_aliases()

    print("\n" + "=" * 60)
    print("Tests completed. Run 'py main.py' to start the full market analysis.")
