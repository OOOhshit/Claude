# Oil Companies Market & Technology Scanner

This project automatically scans major oil companies' websites to analyze their market presence and technology focus areas. It uses web scraping combined with AI-powered content analysis to generate comprehensive reports.

## Features

- **Automated Web Scraping**: Intelligently finds and scrapes relevant pages about technology, innovation, and market presence
- **AI-Powered Analysis**: Advanced content analysis to identify market segments, technologies, and strategic focus areas
- **Comprehensive Reporting**: Generates multiple output formats (JSON, HTML, CSV) for different analysis needs
- **Respectful Scraping**: Implements delays and respects robots.txt to avoid overwhelming target websites

## Companies Analyzed

- **TotalEnergies** (France)
- **Shell** (Netherlands/UK)  
- **BP** (UK)
- **Chevron** (USA)

## Installation

1. Ensure you have Python 3.7+ installed
2. Install required dependencies:

```bash
py -m pip install -r requirements.txt
```

## Usage

### Basic Usage

Simply run the main script:

```bash
py main.py
```
### Advanced Usage
```bash
py main.py --no-playwright           # Requests only (lighter, faster)
py main.py --proxy-file proxies.txt  # Enable proxy rotation
py main.py --no-resume               # Start fresh, ignore checkpoint
py main.py --no-ssl-verify           # For corporate proxies
```
### Output Files

The scanner generates several output files:

1. **`scraped_content.json`** - Raw scraped data from all websites
2. **`ai_analysis_results.json`** - Detailed AI analysis with confidence scores
3. **`analysis_report.html`** - Interactive HTML report (open in browser)
4. **`company_summary.csv`** - Summary data for spreadsheet analysis
5. **`scanner.log`** - Execution log with detailed information

### Customization

#### Adding New Companies

Edit `companies.json` to add new companies:

```json
{
  "name": "Company Name",
  "url": "https://www.company.com",
  "country": "Country",
  "description": "Brief description"
}
```

#### Modifying Analysis Parameters

Edit the keyword dictionaries in `ai_analyzer.py`:

- **Market Keywords**: Define market segments to identify
- **Technology Keywords**: Define technologies to track
- **Geographic Indicators**: Regions to monitor for presence

## Analysis Methodology

### Web Scraping Strategy

1. **Link Discovery**: Identifies relevant pages using keyword-based filtering
2. **Content Extraction**: Extracts main content while filtering out navigation and ads
3. **Categorization**: Automatically categorizes pages (technology, market, research, etc.)
4. **Rate Limiting**: Implements respectful delays between requests

### AI Analysis Components

1. **Market Segment Analysis**
   - Upstream (exploration, drilling, production)
   - Downstream (refining, petrochemicals, retail)
   - Renewable Energy (solar, wind, hydrogen)
   - Natural Gas (LNG, pipeline, distribution)
   - Chemicals (petrochemicals, plastics, polymers)
   - Carbon Management (CCUS, net zero initiatives)

2. **Technology Identification**
   - Digital Technologies (AI, machine learning, analytics)
   - Automation (robotics, autonomous operations)
   - IoT & Sensors (monitoring, smart systems)
   - Subsea Technology (deepwater, underwater equipment)
   - Carbon Technologies (capture, utilization, storage)
   - Renewable Technologies (solar, wind, energy storage)

3. **Scoring Metrics**
   - **Sustainability Focus**: Based on mentions of renewable energy, carbon neutrality, etc.
   - **Innovation Score**: Based on R&D, breakthrough technologies, patents
   - **Geographic Presence**: Identified regions and countries of operation

## Sample Output

### Market Segments (with confidence scores)
- Upstream: 85% confidence
- Renewable Energy: 72% confidence  
- Carbon Management: 68% confidence

### Technologies Identified
- Digital Technologies: AI, machine learning, data analytics
- Automation: Robotics, autonomous operations
- Carbon Tech: CCUS, carbon utilization

### Sustainability & Innovation Metrics
- Sustainability Focus: 78%
- Innovation Score: 65%

## Technical Details

### Dependencies
- `requests`: HTTP requests and web scraping
- `beautifulsoup4`: HTML parsing and content extraction
- `lxml`: Fast XML/HTML processing
- `urllib3`: URL handling utilities

### Rate Limiting
- 1-2 second delays between page requests
- 2 second delays between companies
- Maximum 10 pages per company to respect server resources

### Error Handling
- Network timeout handling
- Graceful degradation when pages are inaccessible
- Comprehensive logging for troubleshooting

## Ethical Considerations

- **Respectful Scraping**: Implements delays and limits to avoid server overload
- **Public Information Only**: Only scrapes publicly available information
- **No Personal Data**: Focuses on corporate and technical information only
- **Attribution**: Maintains source URLs for transparency

## Limitations

- Analysis is based on publicly available web content only
- Results depend on website structure and content availability
- Some technical details may not be captured if not prominently featured
- Confidence scores are indicative and based on keyword frequency

## Troubleshooting

### Common Issues

1. **No Content Scraped**: Check internet connectivity and website accessibility
2. **Low Confidence Scores**: Websites may use different terminology than expected
3. **Missing Technologies**: Update keyword lists in `ai_analyzer.py`

### Debugging

Enable detailed logging by checking the `scanner.log` file for:
- Network request details
- Content extraction results
- Analysis processing steps

## Future Enhancements

- Integration with official APIs where available
- Support for additional languages
- Real-time monitoring and change detection
- Integration with external AI services (OpenAI, etc.)
- Competitive analysis features
