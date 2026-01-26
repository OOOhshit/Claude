#!/usr/bin/env python3
"""
Oil Companies Market & Technology Scanner - Main Script

Complete pipeline for scraping oil company websites and analyzing their market presence.
"""

import json
import logging
import sys
from pathlib import Path
from scraper import OilCompanyScanner
from ai_analyzer import AIAnalyzer

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
    """Format evidence list with URLs for HTML display"""
    if not evidence_list:
        return "No evidence found"

    formatted_items = []
    for item in evidence_list:
        if isinstance(item, dict):
            # New format with keyword, mention_count, and urls
            keyword = item.get('keyword', '')
            count = item.get('mention_count', 0)
            urls = item.get('urls', [])

            if urls:
                url_links = ', '.join([f'<a href="{url}" target="_blank">{url[:50]}...</a>' if len(url) > 50 else f'<a href="{url}" target="_blank">{url}</a>' for url in urls[:5]])
                formatted_items.append(f"<strong>{keyword}</strong> ({count}x) - Sources: {url_links}")
            else:
                formatted_items.append(f"<strong>{keyword}</strong> ({count}x)")
        else:
            # Legacy string format
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
        
        sustainability_class = "high-score" if result['sustainability_focus'] > 0.6 else "medium-score" if result['sustainability_focus'] > 0.3 else "low-score"
        innovation_class = "high-score" if result['innovation_score'] > 0.6 else "medium-score" if result['innovation_score'] > 0.3 else "low-score"
        
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
        
        html_content += f"""
            <div class="section">
                <h3>Sustainability & Innovation Metrics</h3>
                <p>Sustainability Focus: <span class="score {sustainability_class}">{result['sustainability_focus']:.1%}</span></p>
                <p>Innovation Score: <span class="score {innovation_class}">{result['innovation_score']:.1%}</span></p>
            </div>

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
    logger.info("Starting Oil Companies Market & Technology Analysis")
    
    try:
        # Step 1: Initialize scanner and run scraping
        logger.info("Step 1: Initializing web scraper...")
        scanner = OilCompanyScanner()
        
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
        
        # Step 2: Initialize AI analyzer and process content
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
        
        # Run AI analysis
        logger.info("Step 4: Running AI analysis...")
        company_profiles = analyzer.analyze_all_companies(scraped_data)

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
                         'Sustainability Score', 'Innovation Score', 'Geographic Reach']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in analysis_results:
                top_markets = ', '.join([seg['name'] for seg in result['market_segments'][:3]])
                top_techs = ', '.join([tech['name'] for tech in result['technologies'][:3]])
                
                writer.writerow({
                    'Company': result['company'],
                    'Top Market Segments': top_markets,
                    'Top Technologies': top_techs,
                    'Sustainability Score': f"{result['sustainability_focus']:.1%}",
                    'Innovation Score': f"{result['innovation_score']:.1%}",
                    'Geographic Reach': len(result['geographic_presence'])
                })
        
        logger.info("Analysis complete! Generated files:")
        logger.info("- scraped_content.json: Raw scraped data")
        logger.info("- visited_urls.json: All URLs visited during scanning")
        logger.info("- ai_analysis_results.json: Detailed AI analysis with source URLs")
        logger.info("- completeness_suggestions.json: Suggestions for improving analysis")
        logger.info("- analysis_report.html: Interactive HTML report")
        logger.info("- company_summary.csv: Summary data for spreadsheet analysis")
        logger.info("- scanner.log: Execution log with all URL visits and errors")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()