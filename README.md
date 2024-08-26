# AI News Report Generator

This project is an automated tool that generates comprehensive reports on the latest developments and impacts of AI across various fields. It fetches recent news articles, processes them using AI to determine relevance and provide summaries, and compiles a bilingual (English and Chinese) report.

## Features

- Fetches recent AI-related news from Google Custom Search API
- Uses OpenAI's GPT model to determine relevance, categorize, and summarize news articles
- Generates bilingual (English and Chinese) summaries for each news item
- Attempts to extract more precise article links from general web pages
- Compiles a markdown report categorized by different aspects of AI development and application

## Prerequisites

Before you begin, ensure you have met the following requirements:

- You have a Python 3.7+ environment
- You have obtained API keys for OpenAI and Google Custom Search

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/ai-news-report-generator.git
   cd ai-news-report-generator
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   Create a `.env` file in the root directory of the project and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   GOOGLE_API_KEY=your_google_api_key
   GOOGLE_CX=your_google_custom_search_engine_id
   ```

## Usage

To run the script and generate a report:

```
python main.py
```

The generated report will be saved in the `reports` directory.

## Configuration

You can modify the search queries and categories in the `main.py` file to customize the type of news the tool fetches and how it categorizes them.

## Contributing

Contributions to this project are welcome. Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- OpenAI for providing the GPT model used in this project
- Google for their Custom Search API

## Disclaimer

This tool is for educational and research purposes only. Please respect the terms of service of the APIs used and ensure you have the right to use and process the news content as implemented in this tool.