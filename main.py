#!/usr/bin/env python3
"""
Oil Companies Market & Technology Scanner - Main Script

Complete pipeline for scraping oil company websites and analyzing their market presence.
"""

import json
import logging
import sys
import argparse
from pathlib import Path
from scraper import OilCompanyScanner
from ai_analyzer import AIAnalyzer
from annual_report_fetcher import AnnualReportFetcher
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scanner.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def format_evidence_with_urls(evidence_list):
    """Format evidence list with URLs for HTML display.

    Supports both the new contextual format (keyword, context, sentiment, urls)
    and the legacy format (keyword, mention_count, urls).
    """
    if not evidence_list:
        return "No evidence found"

    formatted_items = []
    for item in evidence_list:
        if isinstance(item, dict):
            keyword = item.get('keyword', '')
            urls = item.get('urls', [])
            context = item.get('context', '')
            sentiment = item.get('sentiment', '')
            count = item.get('mention_count', 0)

            # Build the display line
            parts = [f"<strong>{keyword}</strong>"]

            # Show sentiment badge if available
            if sentiment == 'positive':
                parts.append('<span style="color:#28a745;font-size:0.85em;">[active engagement]</span>')
            elif sentiment == 'negative':
                parts.append('<span style="color:#dc3545;font-size:0.85em;">[negative]</span>')

            # Show count if available (legacy format)
            if count and not context:
                parts.append(f"({count}x)")

            # Show context snippet if available (new format)
            if context:
                # Truncate context for display
                display_ctx = context[:150] + "..." if len(context) > 150 else context
                parts.append(f'<span style="color:#555;font-size:0.85em;">— "{display_ctx}"</span>')

            # Show source URLs
            if urls:
                url_links = ', '.join([
                    f'<a href="{url}" target="_blank">{url[:50]}...</a>' if len(url) > 50
                    else f'<a href="{url}" target="_blank">{url}</a>'
                    for url in urls[:5]
                ])
                parts.append(f"<br><span style='font-size:0.8em;color:#888;'>Sources: {url_links}</span>")

            formatted_items.append(" ".join(parts))
        else:
            formatted_items.append(str(item))

    return "<br>".join(formatted_items)


def create_detailed_report(analysis_results, completeness_suggestions=None):
    """Create a detailed HTML report"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Oil Companies Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .company { margin-bottom: 40px; border: 1px solid #ccc; padding: 20px; border-radius: 8px; }
            .company h2 { color: #2c5aa0; margin-top: 0; }
            .section { margin: 20px 0; }
            .section h3 { color: #444; }
            .market-segment, .technology { 
                background: #f5f5f5; margin: 5px 0; padding: 10px; border-radius: 4px; 
            }
            .new-market-opportunity {
                background: linear-gradient(135deg, #fff3cd, #ffeaa7); 
                margin: 5px 0; padding: 15px; border-radius: 6px;
                border-left: 4px solid #f39c12;
                position: relative;
            }
            .new-market-opportunity::before {
                content: "🚨 NEW OPPORTUNITY";
                position: absolute;
                top: 5px;
                right: 10px;
                background: #e74c3c;
                color: white;
                padding: 2px 8px;
                border-radius: 10px;
                font-size: 0.7em;
                font-weight: bold;
            }
            .confidence { font-weight: bold; color: #007700; }
            .low-confidence { color: #cc7700; }
            .evidence { font-size: 0.9em; color: #666; }
            .score { font-size: 1.2em; font-weight: bold; }
            .high-score { color: #007700; }
            .medium-score { color: #cc7700; }
            .low-score { color: #cc0000; }
            .completeness-section { background: #f0f8ff; border: 1px solid #007acc; padding: 15px; border-radius: 8px; margin-top: 20px; }
            .completeness-score { font-size: 1.3em; font-weight: bold; }
            .score-high { color: #28a745; }
            .score-medium { color: #ffc107; }
            .score-low { color: #dc3545; }
            .suggestion-item { background: #fff; padding: 10px; margin: 5px 0; border-left: 3px solid #007acc; }
            .suggestion-high { border-left-color: #dc3545; }
            .suggestion-medium { border-left-color: #ffc107; }
            .suggestion-low { border-left-color: #28a745; }
            .verification-step { background: #e8f4f8; padding: 10px; margin: 5px 0; border-radius: 4px; }
            .step-title { font-weight: bold; color: #2c5aa0; }
            .annual-report-section { background: linear-gradient(135deg, #e8f5e9, #c8e6c9); border: 2px solid #4caf50; padding: 20px; border-radius: 8px; margin-top: 20px; }
            .annual-report-section h3 { color: #2e7d32; margin-top: 0; }
            .ar-badge { background: #4caf50; color: white; padding: 3px 10px; border-radius: 12px; font-size: 0.8em; font-weight: bold; }
            .strategic-priority { background: #fff; padding: 10px; margin: 5px 0; border-left: 4px solid #2196f3; border-radius: 4px; }
            .tech-investment { background: #fff; padding: 10px; margin: 5px 0; border-left: 4px solid #9c27b0; border-radius: 4px; }
            .tech-investment.strong { border-left-color: #e91e63; }
            .market-expansion { background: #fff; padding: 10px; margin: 5px 0; border-left: 4px solid #ff9800; border-radius: 4px; }
            .future-outlook { background: #e3f2fd; padding: 10px; margin: 5px 0; border-radius: 4px; }
            .risk-factor { background: #ffebee; padding: 5px 10px; margin: 3px 0; border-radius: 4px; display: inline-block; font-size: 0.9em; }
            .partnership { background: #f3e5f5; padding: 8px; margin: 3px 0; border-radius: 4px; }
        </style>
    </head>
    <body>
        <h1>Oil Companies Market & Technology Analysis Report</h1>
        <p>Generated on: <span id="date"></span></p>
        <script>document.getElementById('date').textContent = new Date().toLocaleString();</script>
    """
    
    for result in analysis_results:
        confidence_class = "high-confidence" if len(result.get('market_segments', [])) > 2 else "low-confidence"
        
        html_content += f"""
        <div class="company">
            <h2>{result['company']}</h2>
            
            <div class="section">
                <h3>Executive Summary</h3>
                <p>{result['summary']}</p>
            </div>
            
            <div class="section">
                <h3>Market Segments</h3>
        """
        
        for segment in result.get('market_segments', []):
            conf_score = segment['confidence']
            conf_class = "high-confidence" if conf_score > 0.7 else "medium-confidence" if conf_score > 0.4 else "low-confidence"
            # Format evidence with URLs
            evidence_html = format_evidence_with_urls(segment['evidence'])
            html_content += f"""
                <div class="market-segment">
                    <strong>{segment['name']}</strong>
                    <span class="confidence {conf_class}">({conf_score:.1%} confidence)</span>
                    <div class="evidence">{evidence_html}</div>
                </div>
            """
        
        html_content += """
            </div>
            
            <div class="section">
                <h3>Technologies</h3>
        """
        
        for tech in result.get('technologies', []):
            conf_score = tech['confidence']
            conf_class = "high-confidence" if conf_score > 0.7 else "medium-confidence" if conf_score > 0.4 else "low-confidence"
            # Format evidence with URLs
            evidence_html = format_evidence_with_urls(tech['evidence'])
            html_content += f"""
                <div class="technology">
                    <strong>{tech['name']}</strong>
                    <span style="background: #e0e0e0; padding: 2px 6px; border-radius: 3px; font-size: 0.8em;">{tech['category']}</span>
                    <span class="confidence {conf_class}">({conf_score:.1%} confidence)</span>
                    <div class="evidence">{evidence_html}</div>
                </div>
            """
        
        # Add new market opportunities section
        html_content += """
            </div>
        """
        
        if result.get('new_market_opportunities'):
            html_content += """
            <div class="section">
                <h3>🚨 New Market Opportunities</h3>
            """
            for opportunity in result['new_market_opportunities']:
                conf_score = opportunity['confidence']
                conf_class = "high-confidence" if conf_score > 0.6 else "medium-confidence" if conf_score > 0.3 else "low-confidence"
                # Format evidence with URLs
                evidence_html = format_evidence_with_urls(opportunity['evidence'])
                html_content += f"""
                    <div class="new-market-opportunity">
                        <strong>{opportunity['name']}</strong>
                        <span style="background: #3498db; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.8em; margin-left: 10px;">{opportunity['potential_category']}</span>
                        <span class="confidence {conf_class}">({conf_score:.1%} confidence)</span>
                        <div class="evidence">{evidence_html}</div>
                    </div>
                """
            html_content += """
            </div>
            """

        # Add Business Activities Section
        if result.get('business_activities'):
            html_content += """
            <div class="section">
                <h3>Business Activities (Partnerships, Investments, JVs)</h3>
            """
            # Group by activity type
            activity_groups = {}
            for activity in result['business_activities']:
                atype = activity['activity_type'].replace('_', ' ').title()
                if atype not in activity_groups:
                    activity_groups[atype] = []
                activity_groups[atype].append(activity)

            for atype, acts in activity_groups.items():
                html_content += f"<h4>{atype} ({len(acts)})</h4>"
                for act in acts:
                    partners_str = ', '.join(act.get('partners', []))
                    focus = act.get('focus_area', 'General')
                    focus_color = '#27ae60' if focus != 'General' else '#95a5a6'
                    evidence_urls = []
                    for ev in act.get('evidence', []):
                        evidence_urls.extend(ev.get('urls', []))
                    urls_html = ''
                    if evidence_urls:
                        urls_html = ' | '.join([f'<a href="{u}" target="_blank">source</a>' for u in evidence_urls[:3]])
                    html_content += f"""
                    <div style="margin: 8px 0; padding: 8px 12px; background: #f8f9fa; border-left: 3px solid {focus_color}; border-radius: 4px;">
                        <strong>{partners_str}</strong>
                        <span style="background: {focus_color}; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.8em; margin-left: 10px;">{focus}</span>
                        <div style="font-size: 0.85em; color: #555; margin-top: 4px;">{act.get('description', '')}</div>
                        <div style="font-size: 0.8em; color: #888; margin-top: 2px;">{urls_html}</div>
                    </div>
                    """
            html_content += """
            </div>
            """

        # Add Annual Report Analysis Section
        if result.get('annual_report_analysis'):
            ar = result['annual_report_analysis']
            html_content += f"""
            <div class="annual-report-section">
                <h3><span class="ar-badge">ANNUAL REPORT {ar.get('year', 'N/A')}</span> Strategic Analysis</h3>
                <p><em>Source: <a href="{ar.get('source_url', '#')}" target="_blank">{ar.get('source_url', 'N/A')[:80]}...</a></em></p>
            """

            # Strategic Priorities
            if ar.get('strategic_priorities'):
                html_content += "<h4>Strategic Priorities</h4>"
                for priority in ar['strategic_priorities'][:5]:
                    conf = priority.get('confidence', 0)
                    html_content += f"""
                    <div class="strategic-priority">
                        <strong>{priority.get('name', 'N/A')}</strong>
                        <span style="background: #e3f2fd; padding: 2px 6px; border-radius: 3px; font-size: 0.8em; margin-left: 5px;">{priority.get('type', '')}</span>
                        <span class="confidence">({conf:.0%} confidence)</span>
                    </div>
                    """

            # Technology Investments
            if ar.get('technology_investments'):
                html_content += "<h4>Technology Investments</h4>"
                for tech in ar['technology_investments'][:6]:
                    signal = tech.get('investment_signal', 'Mentioned')
                    signal_class = 'strong' if signal == 'Strong' else ''
                    html_content += f"""
                    <div class="tech-investment {signal_class}">
                        <strong>{tech.get('name', 'N/A')}</strong>
                        <span style="background: {'#e91e63' if signal == 'Strong' else '#9e9e9e'}; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.8em; margin-left: 5px;">{signal}</span>
                        <span class="confidence">({tech.get('confidence', 0):.0%})</span>
                    </div>
                    """

            # Market Expansions
            if ar.get('market_expansions'):
                html_content += "<h4>Market Expansion Focus</h4>"
                for expansion in ar['market_expansions'][:5]:
                    html_content += f"""
                    <div class="market-expansion">
                        <strong>{expansion.get('market', 'N/A')}</strong> in <strong>{expansion.get('region', 'N/A')}</strong>
                    </div>
                    """

            # Future Outlook
            if ar.get('future_outlook'):
                html_content += "<h4>Future Outlook & Targets</h4>"
                for outlook in ar['future_outlook'][:5]:
                    html_content += f'<div class="future-outlook">{outlook}</div>'

            # Partnerships
            if ar.get('partnerships'):
                html_content += "<h4>Key Partnerships</h4>"
                for partnership in ar['partnerships'][:5]:
                    html_content += f"""
                    <div class="partnership">
                        <strong>{partnership.get('partner', 'N/A')}</strong>
                        <span style="font-size: 0.9em; color: #666;"> - {partnership.get('focus_area', 'General')}</span>
                    </div>
                    """

            # Risk Factors
            if ar.get('risk_factors'):
                html_content += "<h4>Risk Factors Identified</h4><div>"
                for risk in ar['risk_factors']:
                    html_content += f'<span class="risk-factor">{risk}</span> '
                html_content += "</div>"

            html_content += """
            </div>
            """

        html_content += f"""
            <div class="section">
                <h3>Geographic Presence</h3>
                <p>{', '.join(result['geographic_presence']) if result['geographic_presence'] else 'Limited geographic information found'}</p>
            </div>
        """

        # Add completeness suggestions section
        if completeness_suggestions:
            company_suggestions = next((s for s in completeness_suggestions if s['company'] == result['company']), None)
            if company_suggestions:
                score = company_suggestions['completeness_score']
                score_class = "score-high" if score >= 0.7 else "score-medium" if score >= 0.4 else "score-low"
                html_content += f"""
            <div class="completeness-section">
                <h3>Analysis Completeness Check</h3>
                <p>Completeness Score: <span class="completeness-score {score_class}">{score:.0%}</span></p>
                <p>Pages Analyzed: {company_suggestions['pages_analyzed']}</p>
                <p>Overall Assessment: <strong>{company_suggestions.get('overall_assessment', 'N/A')}</strong></p>
                """

                if company_suggestions.get('suggestions'):
                    html_content += "<h4>Suggestions for Improvement:</h4>"
                    for suggestion in company_suggestions['suggestions']:
                        priority = suggestion.get('priority', 'LOW')
                        priority_class = f"suggestion-{priority.lower()}"
                        html_content += f"""
                <div class="suggestion-item {priority_class}">
                    <strong>[{priority}]</strong> {suggestion.get('issue', '')}
                    <br><em>Recommendation:</em> {suggestion.get('recommendation', '')}
                </div>
                        """

                if company_suggestions.get('verification_steps'):
                    html_content += "<h4>Verification Steps to Ensure Completeness:</h4>"
                    for step in company_suggestions['verification_steps']:
                        html_content += f"""
                <div class="verification-step">
                    <span class="step-title">{step.get('step', '')}</span>
                    <p>{step.get('description', '')}</p>
                """
                        if step.get('urls_to_check'):
                            valid_urls = [u for u in step['urls_to_check'] if u]
                            if valid_urls:
                                html_content += f"<p><em>URLs to check:</em> {', '.join(valid_urls)}</p>"
                        if step.get('search_terms'):
                            html_content += f"<p><em>Search terms:</em> {', '.join(step['search_terms'])}</p>"
                        if step.get('note'):
                            html_content += f"<p><em>Note:</em> {step['note']}</p>"
                        html_content += "</div>"

                html_content += """
            </div>
                """

        html_content += """
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    with open('analysis_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info("Detailed HTML report created: analysis_report.html")

def main():
    """Main execution function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Oil Companies Market & Technology Scanner')
    parser.add_argument('--no-ssl-verify', action='store_true',
                        help='Disable SSL certificate verification (for corporate environments with proxy)')
    parser.add_argument('--no-playwright', action='store_true',
                        help='Disable Playwright browser rendering (use requests only)')
    parser.add_argument('--proxy-file', type=str, default=None,
                        help='Path to proxy list file (one proxy per line: http://host:port)')
    parser.add_argument('--no-resume', action='store_true',
                        help='Start fresh instead of resuming from checkpoint')
    args = parser.parse_args()

    logger.info("Starting Oil Companies Market & Technology Analysis")

    try:
        # Step 1: Initialize scanner and run scraping
        logger.info("Step 1: Initializing web scraper...")
        verify_ssl = not args.no_ssl_verify
        scanner = OilCompanyScanner(
            verify_ssl=verify_ssl,
            use_playwright=not args.no_playwright,
            proxy_file=args.proxy_file,
            resume=not args.no_resume,
        )
        
        # Load companies
        companies = scanner.load_companies()
        if not companies:
            logger.error("No companies found in companies.json")
            return
        
        logger.info(f"Found {len(companies)} companies to analyze")
        
        # Run scraping
        logger.info("Step 2: Starting web scraping process...")
        scanner.run_analysis()
        
        # Check if we have scraped content
        if not scanner.scraped_content:
            logger.warning("No content was scraped. Check network connectivity and website accessibility.")
            return
        
        logger.info(f"Successfully scraped {len(scanner.scraped_content)} pages")

        # Step 2b: Fetch Annual Reports
        logger.info("Step 2b: Fetching annual reports...")
        report_fetcher = AnnualReportFetcher(verify_ssl=verify_ssl)

        # Set target year to last year (e.g., 2025 for reports published in 2026)
        target_year = datetime.now().year - 1
        report_fetcher.set_target_year(target_year)

        # Fetch annual reports for all companies
        annual_reports = report_fetcher.fetch_all_company_reports(companies, download_pdfs=True)

        if annual_reports:
            logger.info(f"Successfully fetched {len(annual_reports)} annual reports")
            report_fetcher.save_results()

            # Log summary
            report_summary = report_fetcher.get_report_summary()
            logger.info(f"Annual Report Summary:")
            logger.info(f"  - Total reports: {report_summary['total_reports']}")
            logger.info(f"  - Companies with reports: {report_summary['companies_with_reports']}")
            logger.info(f"  - PDF reports: {report_summary['pdf_reports']}")
            logger.info(f"  - HTML reports: {report_summary['html_reports']}")
            if report_summary.get('quarterly_fallbacks'):
                logger.info(f"  - Quarterly fallbacks used for: {', '.join(report_summary['quarterly_fallbacks'])}")
        else:
            logger.warning("No annual reports were fetched. This may be due to website restrictions or unavailable reports.")

        # Step 3: Initialize AI analyzer and process content
        logger.info("Step 3: Initializing AI analyzer...")
        analyzer = AIAnalyzer()
        
        # Convert scraped content to the format expected by analyzer
        scraped_data = []
        for content in scanner.scraped_content:
            scraped_data.append({
                'company': content.company,
                'url': content.url,
                'title': content.title,
                'content': content.content,
                'page_type': content.page_type
            })
        
        # Convert annual reports to dict format for analysis
        annual_report_data = None
        if annual_reports:
            annual_report_data = [
                {
                    'company': r.company,
                    'year': r.year,
                    'content': r.content,
                    'url': r.url
                }
                for r in annual_reports
            ]

        # Run AI analysis (annual report content is merged into market/tech analysis)
        logger.info("Step 4: Running AI analysis (website + annual/quarterly reports)...")
        company_profiles = analyzer.analyze_all_companies(scraped_data, annual_report_data)

        # Step 4a: Analyze annual reports for strategic insights and merge with company profiles
        logger.info("Step 4a: Analyzing annual reports for strategic insights...")
        if annual_report_data:
            # Analyze all annual reports for strategic priorities, outlook, etc.
            annual_report_analyses = analyzer.analyze_all_annual_reports(annual_report_data)

            # Merge annual report strategic analysis with company profiles
            for profile in company_profiles:
                if profile.company in annual_report_analyses:
                    profile.annual_report_analysis = annual_report_analyses[profile.company]
                    logger.info(f"Merged annual report strategic analysis for {profile.company}")

                    # Update summary to include annual report insights
                    profile.summary = analyzer.generate_summary(profile)

        # Generate completeness suggestions for each company
        logger.info("Step 4b: Generating completeness suggestions...")
        completeness_suggestions = []
        company_scraped_map = {}
        for item in scraped_data:
            company = item.get('company', 'Unknown')
            if company not in company_scraped_map:
                company_scraped_map[company] = []
            company_scraped_map[company].append(item)

        for profile in company_profiles:
            company_data = company_scraped_map.get(profile.company, [])
            suggestions = analyzer.generate_completeness_suggestions(profile.company, company_data, profile)
            completeness_suggestions.append(suggestions)

        # Save completeness suggestions
        with open('completeness_suggestions.json', 'w', encoding='utf-8') as f:
            json.dump(completeness_suggestions, f, indent=2, ensure_ascii=False)
        logger.info("Completeness suggestions saved to completeness_suggestions.json")

        # Save AI analysis results
        analysis_results = analyzer.save_analysis(company_profiles)
        
        # Step 3: Create comprehensive reports
        logger.info("Step 5: Generating reports...")
        
        # Create detailed HTML report
        create_detailed_report(analysis_results, completeness_suggestions)
        
        # Create summary CSV for easy data analysis
        import csv
        with open('company_summary.csv', 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Company', 'Top Market Segments', 'Top Technologies',
                         'Geographic Reach',
                         'Annual Report Year', 'Strategic Priorities', 'Tech Investments', 'Market Expansions']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for result in analysis_results:
                top_markets = ', '.join([seg['name'] for seg in result['market_segments'][:3]])
                top_techs = ', '.join([tech['name'] for tech in result['technologies'][:3]])

                # Extract annual report data if available
                ar = result.get('annual_report_analysis')
                ar_year = ar.get('year', 'N/A') if ar else 'N/A'
                ar_priorities = ', '.join([p['name'] for p in ar.get('strategic_priorities', [])[:3]]) if ar else 'N/A'
                ar_tech = ', '.join([t['name'] for t in ar.get('technology_investments', []) if t.get('investment_signal') == 'Strong'][:3]) if ar else 'N/A'
                ar_expansions = ', '.join([f"{e['market']}/{e['region']}" for e in ar.get('market_expansions', [])[:3]]) if ar else 'N/A'

                writer.writerow({
                    'Company': result['company'],
                    'Top Market Segments': top_markets,
                    'Top Technologies': top_techs,
                    'Geographic Reach': len(result['geographic_presence']),
                    'Annual Report Year': ar_year,
                    'Strategic Priorities': ar_priorities,
                    'Tech Investments': ar_tech,
                    'Market Expansions': ar_expansions
                })
        
        logger.info("Analysis complete! Generated files:")
        logger.info("- scraped_content.json: Raw scraped data")
        logger.info("- visited_urls.json: All URLs visited during scanning")
        logger.info("- ai_analysis_results.json: Detailed AI analysis with source URLs")
        logger.info("- completeness_suggestions.json: Suggestions for improving analysis")
        logger.info("- analysis_report.html: Interactive HTML report")
        logger.info("- company_summary.csv: Summary data for spreadsheet analysis")
        logger.info("- scanner.log: Execution log with all URL visits and errors")
        logger.info("- annual_reports_content.json: Extracted annual report content")
        logger.info("- annual_reports_search.json: Annual report search results")
        logger.info("- annual_reports/: Downloaded PDF annual reports")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()