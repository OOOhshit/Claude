# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Oil Companies Market & Technology Scanner - A web scraping and AI analysis system to scan major oil companies' websites for their market presence and technology focus areas.

## Development Environment

**Operating System**: Windows  
**Python Command**: Use `py` instead of `python` or `python3`  
**Command Compatibility**: Ensure all commands are Windows-compatible

## Common Commands

### Development Commands
```bash
# Install dependencies (Windows)
py -m pip install -r requirements.txt

# Run the main scanner
py main.py

# Run tests
py test_scanner.py

# Check Python version
py --version
```

### Alternative Windows Execution
```bash
# Use Windows batch file for easy execution
run_scanner.bat
```

## Architecture

### Core Components
- **scraper.py**: Web scraping engine with BeautifulSoup and requests
- **ai_analyzer.py**: AI-powered content analysis and market intelligence
- **main.py**: Main orchestration script combining scraping and analysis
- **companies.json**: Target company list with official URLs

### Analysis Pipeline
1. **Web Scraping**: Intelligent link discovery and content extraction
2. **Content Categorization**: Automatic page type classification
3. **AI Analysis**: Market segments, technologies, and metrics extraction
4. **Report Generation**: Multiple output formats (HTML, JSON, CSV)

### Output Files
- `scraped_content.json`: Raw scraped data
- `ai_analysis_results.json`: Detailed AI analysis with confidence scores
- `analysis_report.html`: Interactive HTML report
- `company_summary.csv`: Spreadsheet-friendly summary
- `scanner.log`: Execution logs

## Key Features

### Market Segments Tracked
- Upstream (exploration, drilling, production)
- Downstream (refining, petrochemicals, retail)
- Renewable Energy (solar, wind, hydrogen)
- Natural Gas (LNG, pipeline, distribution)
- Chemicals (petrochemicals, plastics)
- Carbon Management (CCUS, net zero)

### Technologies Monitored
- Digital Technologies (AI, ML, analytics)
- Automation (robotics, autonomous operations)
- IoT & Sensors (monitoring, smart systems)
- Subsea Technology (deepwater equipment)
- Carbon Technologies (capture, storage)
- Renewable Technologies (solar, wind, storage)

### Scoring Metrics
- **Sustainability Focus**: Based on renewable energy and carbon neutrality mentions
- **Innovation Score**: R&D, patents, and breakthrough technology indicators
- **Geographic Presence**: Identified regions and countries of operation

## Development Guidelines

### Windows-Specific Commands
- Always use `py` command instead of `python` or `python3`
- Use forward slashes or escaped backslashes in file paths
- Test batch files for Windows users (`run_scanner.bat`)

### Code Patterns
- **Error Handling**: Comprehensive exception handling for network requests
- **Rate Limiting**: Respectful 1-2 second delays between requests
- **Content Filtering**: Remove scripts, styles, and navigation elements
- **Confidence Scoring**: Evidence-based scoring for all analysis results

### Adding New Companies
Edit `companies.json`:
```json
{
  "name": "Company Name",
  "url": "https://www.company.com",
  "country": "Country",
  "description": "Brief description"
}
```

### Customizing Analysis
Modify keyword dictionaries in `ai_analyzer.py`:
- `market_keywords`: Define market segments to identify
- `technology_keywords`: Define technologies to track
- `geographic_indicators`: Regions to monitor

## Dependencies

Core Python packages:
- `requests`: HTTP requests and web scraping
- `beautifulsoup4`: HTML parsing and content extraction
- `lxml`: Fast XML/HTML processing
- `urllib3`: URL handling utilities

## Testing

Run comprehensive tests:
```bash
py test_scanner.py
```

Tests include:
- Company data loading validation
- AI analyzer functionality verification
- Sample content analysis demonstration

## Ethical Considerations

- **Respectful Scraping**: 1-2 second delays, 10 page limit per site
- **Public Information Only**: No private or restricted content
- **Attribution**: Maintains source URLs for transparency
- **Rate Limiting**: Prevents server overload