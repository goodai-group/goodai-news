# AI News Report Generator

This project is an automated tool that generates comprehensive reports on the latest developments and impacts of AI across various fields. It fetches recent news articles from multiple sources, processes them using AI to determine relevance and provide summaries, and compiles a bilingual (English and Chinese) report.

## Features

- Fetches AI news from multiple sources including RSS feeds and web scraping
- Filters news articles based on recency
- Uses Google's Gemini AI to process and summarize news articles
- Generates bilingual (English and Chinese) summaries and comments
- Categorizes news articles into relevant AI fields
- Produces a comprehensive markdown report

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7+
- Google API key for Gemini AI

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ai-news-report-generator.git
   cd ai-news-report-generator
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   Create a `.env` file in the root directory of the project and add your API key:
   ```
   GOOGLE_API_KEY=your_google_api_key
   ```

## Usage

Run the main script to generate the report:

```
python main.py
```

The script will fetch news, process it, and generate a report in the `reports` directory.

## Configuration

You can modify the `ai_news_sources` list in the `main.py` file to add or remove news sources. Each source should have a name, URL, and type (either 'rss' or 'web').

You can also adjust the following parameters in the `main.py` file:

- `limit_per_source`: Maximum

## Contributing

Contributions to this project are welcome. Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Google for providing the Gemini AI model used in this project
- Various news sources for their RSS feeds and web content

## Disclaimer

This tool is for educational and research purposes only. Please respect the terms of service of the APIs used and ensure you have the right to use and process the news content as implemented in this tool.