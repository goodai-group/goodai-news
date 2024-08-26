import os
import requests
from openai import OpenAI
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get API keys from environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')
google_cx = os.getenv('GOOGLE_CX')

client = OpenAI(api_key=openai_api_key)

def get_news(search_query, api_key, cx, num=10):
    date_restrict = f"date:r:{int((datetime.now() - timedelta(days=30)).timestamp())}:{int(datetime.now().timestamp())}"
    url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cx}&q={search_query}&num={num}&dateRestrict={date_restrict}&sort=date"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get('items', [])
    except requests.RequestException as e:
        logging.error(f"Error fetching news for query '{search_query}': {e}")
        return []

def extract_article_link(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        article = soup.find('article') or soup.find('div', class_='article')
        if article:
            link = article.find('a')
            if link and link.get('href'):
                return link['href']
    except Exception as e:
        logging.error(f"Error extracting article link from {url}: {e}")
    return url

def check_relevance_and_process(title, snippet, link):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI relevance checker, summarizer, and translator. First, determine if the news is relevant to recent AI developments or impacts. If relevant, provide a concise summary in English and Chinese, and categorize the news into one of these categories: 'Core AI Progress', 'AI in Research', 'AI in Finance', 'AI in Education', 'AI in Military', 'Beneficial AI Applications', or 'Other'. If not relevant, simply respond with 'Not relevant'. Format the output as follows:\nRelevance: [Yes/No]\nCategory: [Category]\n[English title]\n[Chinese title]\n[English summary]\n[Chinese summary]"},
                {"role": "user", "content": f"Title: {title}\n\nSnippet: {snippet}\n\nLink: {link}"}
            ],
            max_tokens=300,
            n=1,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error in relevance check and processing: {e}")
        return None

def generate_report(news_data, output_dir="reports"):
    if not news_data:
        logging.warning("No news data provided.")
        return

    report = "# 最新AI进展与影响周报\n\n"
    current_date = datetime.now().strftime("%Y-%m-%d")
    filename = f"ai_progress_report_{current_date}.md"

    categories = {
        'Core AI Progress': [],
        'AI in Research': [],
        'AI in Finance': [],
        'AI in Education': [],
        'AI in Military': [],
        'Beneficial AI Applications': [],
        'Other': []
    }

    for news in news_data:
        title = news.get('title', '无标题')
        snippet = news.get('snippet', '无摘要')
        link = news.get('link', '#')
        
        article_link = extract_article_link(link)

        processed_content = check_relevance_and_process(title, snippet, article_link)
        if processed_content and not processed_content.startswith("Not relevant"):
            try:
                lines = processed_content.split('\n')
                relevance = lines[0].split(': ')[1]
                if relevance.lower() == 'yes':
                    category = lines[1].split(': ')[1]
                    en_title, cn_title, en_summary, cn_summary = lines[2:]
                    
                    clickable_title = f"### [{en_title}]({article_link})\n{cn_title}\n\n"
                    content = f"{en_summary}\n\n{cn_summary}\n\n"
                    
                    categories.get(category, categories['Other']).append(f"{clickable_title}{content}")
            except Exception as e:
                logging.error(f"Error processing news item: {e}")
                logging.debug(f"Problematic content: {processed_content}")

        time.sleep(1)  # Add delay to avoid exceeding API rate limits

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
    search_queries = [
        "AI progress",
        "AI research breakthroughs",
        "AI in finance",
        "AI in education",
        "AI military applications",
        "beneficial AI applications"
    ]

    all_news = []
    for query in search_queries:
        news = get_news(query, google_api_key, google_cx, num=5)  # 每个查询获取5条新闻
        all_news.extend(news)
        time.sleep(2)  # 在查询之间添加延迟

    if not all_news:
        logging.warning("No news found. Exiting.")
        return

    generate_report(all_news)

if __name__ == "__main__":
    main()