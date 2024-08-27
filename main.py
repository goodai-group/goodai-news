import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
import logging
from dotenv import load_dotenv
import hashlib
import feedparser
import concurrent.futures
from urllib.parse import urljoin
from dateutil import parser
import pytz
import google.generativeai as genai

# Load environment variables and configure logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get API key from environment variables
google_api_key = os.getenv('GOOGLE_API_KEY')

# Configure Gemini
genai.configure(api_key=google_api_key)

# List of AI news sources
ai_news_sources = [
    {"name": "MIT Technology Review - AI", "url": "https://www.technologyreview.com/topic/artificial-intelligence/", "type": "web"},
    {"name": "AI News by Synced", "url": "https://syncedreview.com/feed/", "type": "rss"},
    {"name": "Arxiv AI Papers", "url": "http://arxiv.org/rss/cs.AI", "type": "rss"},
    {"name": "Google AI Blog", "url": "https://blog.google/technology/ai/rss/", "type": "rss"},
    {"name": "DeepMind Blog", "url": "https://deepmind.com/blog/feed/basic/", "type": "rss"},
    {"name": "OpenAI Blog", "url": "https://openai.com/blog/rss/", "type": "rss"},
    {"name": "AI Trends", "url": "https://www.aitrends.com/feed/", "type": "rss"},
    {"name": "Towards Data Science - AI", "url": "https://towardsdatascience.com/feed/tagged/artificial-intelligence", "type": "rss"}
]

def parse_date(date_str):
    """Parse a date string to a datetime object with UTC timezone."""
    try:
        return parser.parse(date_str).replace(tzinfo=pytz.UTC)
    except Exception as e:
        logging.warning(f"Could not parse date: {date_str}. Error: {e}")
        return None

def parse_rss_feed(source):
    """Parse an RSS feed and extract relevant information."""
    try:
        feed = feedparser.parse(source['url'])
        articles = []
        for entry in feed.entries:
            published = entry.get('published', '')
            articles.append({
                'title': entry.get('title', ''),
                'summary': entry.get('summary', ''),
                'link': entry.get('link', ''),
                'published': parse_date(published) if published else None,
                'source': source['name']
            })
        return articles
    except Exception as e:
        logging.error(f"Error parsing RSS feed {source['name']}: {e}")
        return []

def scrape_mit_tech_review(source):
    """Scrape articles from MIT Technology Review AI section."""
    try:
        response = requests.get(source['url'])
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = []
        for article in soup.select('div.gsFOMIMb'):
            title_elem = article.select_one('h3 a')
            if title_elem:
                title = title_elem.text.strip()
                link = urljoin(source['url'], title_elem['href'])
                summary = article.select_one('p.OctaONyK').text.strip() if article.select_one('p.OctaONyK') else ''
                articles.append({
                    'title': title,
                    'summary': summary,
                    'link': link,
                    'published': None,  # MIT Tech Review doesn't provide easy date extraction
                    'source': source['name']
                })
        return articles
    except Exception as e:
        logging.error(f"Error scraping {source['name']}: {e}")
        return []

def get_news_from_source(source):
    """Get news from a specific source based on its type."""
    if source['type'] == 'rss':
        return parse_rss_feed(source)
    elif source['name'] == "MIT Technology Review - AI":
        return scrape_mit_tech_review(source)
    else:
        logging.warning(f"Unsupported source type for {source['name']}")
        return []

def get_all_ai_news():
    """Fetch news from all defined AI news sources concurrently."""
    all_news = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_source = {executor.submit(get_news_from_source, source): source for source in ai_news_sources}
        for future in concurrent.futures.as_completed(future_to_source):
            source = future_to_source[future]
            try:
                news = future.result()
                all_news.extend(news)
                logging.info(f"Retrieved {len(news)} articles from {source['name']}")
            except Exception as e:
                logging.error(f"Error retrieving news from {source['name']}: {e}")
    return all_news

def filter_recent_news(news_list, days=1):
    """Filter news articles to include only those published within the specified number of days."""
    recent_news = []
    cutoff_date = datetime.now(pytz.UTC) - timedelta(days=days)
    for news in news_list:
        if news['published']:
            if news['published'] > cutoff_date:
                recent_news.append(news)
        else:
            recent_news.append(news)  # If no date, include it anyway
    return recent_news

def generate_content_hash(title, summary):
    """Generate a hash of the content for duplicate detection."""
    content = f"{title}{summary}".encode('utf-8')
    return hashlib.md5(content).hexdigest()

def batch_process_news(news_list, batch_size=5):
    """Process news in batches using the Gemini model."""
    processed_news = []
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    for i in range(0, len(news_list), batch_size):
        batch = news_list[i:i+batch_size]
        batch_content = "\n\n".join([f"Title: {news['title']}\nSummary: {news['summary']}\nLink: {news['link']}" for news in batch])
        
        prompt = f"""For each news item, determine if it's relevant to recent AI developments or impacts. 
        If relevant, provide a concise summary in English and Chinese, and categorize the news into one of these categories: 
        'AI Breakthroughs and Innovations', 'Machine Learning and Deep Learning', 'AI in Finance and Fintech', 
        'AI in Education and EdTech', 'AI in Healthcare and Medicine', 'AI for Social Good and Environment', 
        'AI in Business and Industry', 'AI Ethics, Governance, and Policy', 'AI in Defense and Security', 
        'Emerging AI Technologies', or 'Other AI Developments'.
        If not relevant, simply respond with 'Not relevant'. 
        Format the output as follows for each news item:
        Relevance: [Yes/No]
        Category: [Category]
        English Title: [English title]
        Chinese Title: [Chinese title]
        English Summary: [English summary]
        Chinese Summary: [Chinese summary]
        ---
        
        Process the following news items:

        {batch_content}
        """
        
        try:
            response = model.generate_content(prompt)
            batch_result = response.text.split('---')
            for news, result in zip(batch, batch_result):
                if not result.strip().startswith("Not relevant"):
                    processed_news.append({
                        'original': news,
                        'processed': result.strip()
                    })
            logging.info(f"Processed batch of {len(batch)} news items")
        except Exception as e:
            logging.error(f"Error processing batch: {e}")
        
        time.sleep(2)  # Add delay between batches to avoid rate limiting
    
    return processed_news

def generate_report(news_data, output_dir="reports"):
    """Generate a report from the processed news data."""
    if not news_data:
        logging.warning("No news data provided.")
        return

    report = "# Latest AI Progress and Impact Daily Report\n\n"
    current_date = datetime.now().strftime("%Y-%m-%d")
    filename = f"ai_progress_report_{current_date}.md"

    categories = {
        'AI Breakthroughs and Innovations': [],
        'Machine Learning and Deep Learning': [],
        'AI in Finance and Fintech': [],
        'AI in Education and EdTech': [],
        'AI in Healthcare and Medicine': [],
        'AI for Social Good and Environment': [],
        'AI in Business and Industry': [],
        'AI Ethics, Governance, and Policy': [],
        'AI in Defense and Security': [],
        'Emerging AI Technologies': [],
        'Other AI Developments': []
    }

    for item in news_data:
        original = item['original']
        processed = item['processed']
        
        try:
            lines = processed.split('\n')
            relevance = lines[0].split(': ')[1]
            if relevance.lower() == 'yes':
                category = lines[1].split(': ')[1]
                en_title = lines[2].split(': ', 1)[1]
                cn_title = lines[3].split(': ', 1)[1]
                en_summary = lines[4].split(': ', 1)[1]
                cn_summary = lines[5].split(': ', 1)[1]
                
                clickable_title = f"### [{en_title}]({original['link']})\n{cn_title}\n\n"
                content = f"{en_summary}\n\n{cn_summary}\n\n"
                
                categories.get(category, categories['Other AI Developments']).append(f"{clickable_title}{content}")
        except Exception as e:
            logging.error(f"Error processing news item: {e}")
            logging.debug(f"Problematic content: {processed}")

    for category, items in categories.items():
        if items:
            report += f"## {category}\n\n"
            report += ''.join(items)

    output_path = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        logging.info(f"Report generated successfully: {filename}")
    except IOError as e:
        logging.error(f"Error writing report: {e}")

def main():
    """Main function to orchestrate the news gathering and report generation process."""
    all_ai_news = get_all_ai_news()
    recent_ai_news = filter_recent_news(all_ai_news, days=1)
    logging.info(f"Retrieved a total of {len(all_ai_news)} articles, {len(recent_ai_news)} are recent.")
    
    processed_news = batch_process_news(recent_ai_news)
    generate_report(processed_news)

if __name__ == "__main__":
    main()