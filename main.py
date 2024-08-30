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
import re

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
def clean_html(text):
    """Remove HTML tags from a string"""
    clean_text = re.sub('<[^<]+?>', '', text)
    return clean_text.replace('&nbsp;', ' ').strip()

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

def get_news_from_source(source, limit=10):
    """
    Get news from a specific source based on its type, limiting to the most recent items.
    
    Args:
    source (dict): Source information including type and URL.
    limit (int): Maximum number of news items to return per source.
    
    Returns:
    list: List of dictionaries containing article information, limited to the most recent 'limit' items.
    """
    if source['type'] == 'rss':
        news = parse_rss_feed(source)
    elif source['name'] == "MIT Technology Review - AI":
        news = scrape_mit_tech_review(source)
    else:
        logging.warning(f"Unsupported source type for {source['name']}")
        return []
    # Sort news by publication date (if available) and limit to the most recent 'limit' items
    sorted_news = sorted(news, key=lambda x: x['published'] or datetime.min.replace(tzinfo=pytz.UTC), reverse=True)
    return sorted_news[:limit]

def get_all_ai_news(limit_per_source=10):
    """
    Fetch news from all defined AI news sources concurrently, limiting items per source.
    
    Args:
    limit_per_source (int): Maximum number of news items to fetch from each source.
    
    Returns:
    list: Combined list of all fetched news articles, limited per source.
    """
    all_news = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_source = {executor.submit(get_news_from_source, source, limit_per_source): source for source in ai_news_sources}
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
       logging.info(f"Filtering news. Cutoff date: {cutoff_date}")
       
       for news in news_list:
           logging.info(f"Title: {news['title'][:50]}...")
           logging.info(f"Source: {news['source']}")
           logging.info(f"Published date: {news['published']}")
           
           if news['published']:
               if news['published'] > cutoff_date:
                   recent_news.append(news)
                   logging.info("Status: Recent")
               else:
                   logging.info("Status: Old")
           else:
               recent_news.append(news)
               logging.info("Status: No date (included)")
           
           logging.info("---")
       
       logging.info(f"Total news items: {len(news_list)}, Recent news items: {len(recent_news)}")
       return recent_news

def generate_content_hash(title, summary):
    """Generate a hash of the content for duplicate detection."""
    content = f"{title}{summary}".encode('utf-8')
    return hashlib.md5(content).hexdigest()

def batch_process_news(news_list):
    processed_news = []
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    logging.info(f"Starting batch processing of {len(news_list)} news items")
    
    for i, news in enumerate(news_list, 1):
        logging.info(f"Processing news item {i}: {news['title']}")
        prompt = f"""Analyze the following AI news item and provide a relevant comment:

        Title: {news['title']}
        Summary: {news['summary']}
        
        Your task:
        1. Determine if this news is relevant to recent AI developments or impacts.
        2. If relevant, provide a concise summary in English and Chinese, and categorize the news.
        3. Translate the title to Chinese.
        4. If not relevant, respond with 'Not relevant'.
        
        Your response must be directly related to this specific news item.
        Do not include any information not present in the given title and summary.
        Ensure your comment addresses the main points of the article.
        
        Format your response as follows:
        Relevance: [Yes/No]
        Category: [Category]
        English Title: [English title]
        Chinese Title: [Chinese title]
        English Summary: [English summary]
        Chinese Summary: [Chinese summary]
        English Comment: [Brief comment on significance and potential impact]
        Chinese Comment: [Brief comment on significance and potential impact in Chinese]
        """
        
        max_retries = 3
        processed_result = None
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                processed_result = response.text.strip()
                
            except Exception as e:
                logging.error(f"Error processing news item {i}: {e}")
            
            time.sleep(2)  # Add delay between requests to avoid rate limiting
        
        if processed_result:
            lines = processed_result.split('\n')
            relevance = next((line.split(': ', 1)[1] for line in lines if line.startswith('Relevance:')), 'No').lower() == 'yes'
            category = next((line.split(': ', 1)[1] for line in lines if line.startswith('Category:')), '')
            en_title = next((line.split(': ', 1)[1] for line in lines if line.startswith('English Title:')), '')
            cn_title = next((line.split(': ', 1)[1] for line in lines if line.startswith('Chinese Title:')), '')
            en_summary = next((line.split(': ', 1)[1] for line in lines if line.startswith('English Summary:')), '')
            cn_summary = next((line.split(': ', 1)[1] for line in lines if line.startswith('Chinese Summary:')), '')
            en_comment = next((line.split(': ', 1)[1] for line in lines if line.startswith('English Comment:')), '')
            cn_comment = next((line.split(': ', 1)[1] for line in lines if line.startswith('Chinese Comment:')), '')
            
            processed_news.append({
                'original': news,
                'processed': processed_result,
                'relevance': relevance,
                'category': category,
                'en_title': en_title,
                'cn_title': cn_title,
                'en_summary': en_summary,
                'cn_summary': cn_summary,
                'gemini_comment_en': en_comment,
                'gemini_comment_cn': cn_comment,
                'needs_review': not validate_response(news, processed_result)
            })
            logging.info(f"Processed news item {i}" + (" (needs review)" if not validate_response(news, processed_result) else ""))
        else:
            logging.error(f"Failed to generate relevant comment after {max_retries} attempts for article: {news['title']}")
            processed_news.append({
                'original': news,
                'processed': 'Failed to generate comment',
                'relevance': False,
                'category': '',
                'en_title': '',
                'cn_title': '',
                'en_summary': '',
                'cn_summary': '',
                'gemini_comment_en': '',
                'gemini_comment_cn': '',
                'needs_review': True
            })
    
    logging.info(f"Completed batch processing. Processed {len(processed_news)} out of {len(news_list)} news items")
    return processed_news

def validate_response(news, response):
    # Extract the English and Chinese summaries from the response
    lines = response.split('\n')
    en_summary = next((line.split(': ', 1)[1] for line in lines if line.startswith('English Summary:')), '')
    cn_summary = next((line.split(': ', 1)[1] for line in lines if line.startswith('Chinese Summary:')), '')
    
    # Combine both summaries for validation
    combined_summary = en_summary + ' ' + cn_summary
    
    relevant_words = set(news['title'].lower().split() + news['summary'].lower().split())
    response_words = set(combined_summary.lower().split())
    overlap = len(relevant_words.intersection(response_words))
    
    # Check for key phrases or concepts from the article
    key_phrases = [phrase.lower() for phrase in news['title'].split() if len(phrase) > 3]
    key_phrases_present = any(phrase in combined_summary.lower() for phrase in key_phrases)
    
    return overlap > len(relevant_words) * 0.3 and key_phrases_present
   
def generate_report(news_data, output_dir="reports"):
    logging.info(f"Starting report generation with {len(news_data)} news items")
    
    if not news_data:
        logging.warning("No news data provided.")
        return "# Latest AI Progress and Impact Daily Report\n\nNo news items available for today."

    report = "# Latest AI Progress and Impact Daily Report\n\n# 最新人工智能进展与影响日报\n\n"
    current_date = datetime.now().strftime("%Y-%m-%d")
    filename = f"ai_progress_report_{current_date}.md"

    categories = {
        'AI Breakthroughs and Innovations': 'AI 突破与创新',
        'Machine Learning and Deep Learning': '机器学习与深度学习',
        'AI in Finance and Fintech': 'AI 在金融科技中的应用',
        'AI in Education and EdTech': 'AI 在教育科技中的应用',
        'AI in Healthcare and Medicine': 'AI 在医疗保健中的应用',
        'AI for Social Good and Environment': 'AI 促进社会公益与环境保护',
        'AI in Business and Industry': 'AI 在商业和工业中的应用',
        'AI Ethics, Governance, and Policy': 'AI 伦理、治理与政策',
        'AI in Defense and Security': 'AI 在国防与安全中的应用',
        'Emerging AI Technologies': '新兴 AI 技术',
        'Other AI Developments': '其他 AI 发展'
    }

    current_category = ""
    for item in news_data:
        logging.info(f"Processing news item: {item['en_title']}")
        
        if item['relevance']:
            category_en = item['category']
            category_cn = categories.get(category_en, '')
            
            if category_en != current_category:
                if category_cn:
                    report += f"## {category_en}\n\n## {category_cn}\n\n"
                else:
                    report += f"## {category_en}\n\n"
                current_category = category_en
            
            report += f"### [{clean_html(item['en_title'])}]({item['original']['link']})\n\n"
            report += f"### [{clean_html(item['cn_title'])}]({item['original']['link']})\n\n"
            
            report += f"{clean_html(item['en_summary'])}\n\n"
            report += f"{clean_html(item['cn_summary'])}\n\n"
            
            report += f"**AI's Comment:** {clean_html(item['gemini_comment_en'])}\n\n"
            report += f"**AI评论:** {clean_html(item['gemini_comment_cn'])}\n\n"
        else:
            logging.info(f"News item deemed not relevant: {item['en_title']}")

    if not report.strip():
        logging.warning("Generated report is empty")
        return "# Latest AI Progress and Impact Daily Report\n\nNo significant AI news to report today."

    output_path = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        logging.info(f"Report generated successfully: {filename}")
    except IOError as e:
        logging.error(f"Error writing report: {e}")

    logging.info(f"Report content: {report}")
    return report

def select_top_news(news_data, top_n=10):
    if len(news_data) <= top_n:
        logging.info(f"Only {len(news_data)} news items available, using all of them.")
        return news_data

    model = genai.GenerativeModel('gemini-1.5-flash')
    
    all_news_content = "\n\n".join([
        f"Title: {item['en_title']}\nSummary: {item['en_summary']}"
        for item in news_data
    ])
    
    prompt = f"""From the following list of AI news items, select the {top_n} most valuable and impactful ones. 
    For each selected news item, provide its title exactly as it appears in the original list.
    
    Here are the news items:
    
    {all_news_content}
    """
    
    try:
        response = model.generate_content(prompt)
        logging.info(f"Gemini response: {response.text}")
        
        selected_titles = []
        for line in response.text.split('\n'):
            if any(line.strip().startswith(prefix) for prefix in ('**', '-', '*', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.')):
                title = line.split(':', 1)[-1].strip().strip('"')
                selected_titles.append(title)
                logging.info(f"Extracted title: {title}")
        
        top_news = []
        for news in news_data:
            if any(title.lower() in news['en_title'].lower() for title in selected_titles):
                top_news.append(news)
                logging.info(f"Matched news: {news['en_title']}")
        
        logging.info(f"Number of titles extracted: {len(selected_titles)}")
        
        if not selected_titles:
            logging.warning("No titles were extracted from Gemini's response. Using all available news.")
            return news_data
        
        top_news = []
        for news in news_data:
            if any(title.lower() in news['title'].lower() for title in selected_titles):
                top_news.append(news)
                logging.info(f"Matched news: {news['title']}")
        
        if not top_news:
            logging.warning("No matching news items were found. Using all available news.")
            return news_data
        
        logging.info(f"Successfully selected {len(top_news)} top news items")
        return top_news[:top_n]
    except Exception as e:
        logging.error(f"Error selecting top news: {e}")
        logging.info("Using all available news due to selection error.")
        return news_data

def main():
    """Main function to orchestrate the news gathering and report generation process."""
    # Step 1: Retrieve and filter news
    all_ai_news = get_all_ai_news(limit_per_source=100)
    logging.info(f"Retrieved a total of {len(all_ai_news)} articles from all sources.")
    
    recent_ai_news = filter_recent_news(all_ai_news, days=1)
    logging.info(f"Filtered to {len(recent_ai_news)} recent articles (within 1 day).")
    
    # Step 2: Select top 10 most valuable news items
    top_news = select_top_news(recent_ai_news, top_n=10)
    logging.info(f"Selected {len(top_news)} top news items.")
    
    # Step 3: Process selected news items (add Gemini comments)
    processed_news = batch_process_news(top_news)
    logging.info(f"Processed {len(processed_news)} news items with Gemini comments.")
    
    # Step 4 & 5: Generate and format the report
    report = generate_report(processed_news)
    logging.info(f"Generated report with length: {len(report)} characters.")
    
    # You might want to add a function to save the report to a file here

if __name__ == "__main__":
    main()