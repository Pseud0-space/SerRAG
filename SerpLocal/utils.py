import requests 
from bs4 import BeautifulSoup
from urllib.parse import quote
import ollama
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

url_blacklist = ["www.nseindia.com"]

def is_url_blacklisted(url, blacklist):
    for blacklisted_url in blacklist:
        if blacklisted_url in url:
            return True
    return False

def extract_text(url):
    if is_url_blacklisted(url, url_blacklist):
        print(f"BLACKLISTED | URL : {url}")
        return 'skip'
    
    else:
        web = requests.get(url)
        if web.status_code == 200:
            html = web.content
            soup = BeautifulSoup(html, features="html.parser")

            for script in soup(["script", "style"]):
                script.extract()  

            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)

            return text
        
        else:
            return 'skip'

def summaries_web_ollama(model, prompt):
    response = ollama.chat(model=model, messages=[
    {
        'role': 'user',
        'content': prompt,
    },
    ])

    return response['message']['content']

def summaries_web_groq(prompt):
    client = Groq(
        # This is the default and can be omitted
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama2-70b-4096",
    )
    
    return chat_completion.choices[0].message.content

def prompt_template(content, title):
    prompt_template = f"""
    You have extensive experience in summarizing text content extracted from web scraping. Your task is to provide a comprehensive and detailed summary of the given text, ensuring that no important information is omitted. The summary should capture the essence of the content while maintaining the necessary level of detail required for effective retrieval-augmented generation.

    Text content of the webpage after scraping is as follows:

    Title: {title}

    {content}

    In your summary, please adhere to the following guidelines:

    1. Clearly identify the main topic or theme of the text.
    2. Highlight the key points, arguments, or explanations presented in the content.
    3. Preserve relevant factual information, statistics, or data that could be useful for further analysis or generation.
    4. Maintain a logical flow and coherence in the summary, ensuring that the sequence of ideas is easily understandable.
    5. Use concise language and avoid unnecessary repetition or redundancy.
    6. If the text contains technical terminology or domain-specific vocabulary, provide brief explanations or definitions to aid in comprehension.
    If the text includes references to external sources or citations, exclude them from the summary.
    
    Please provide your detailed summary in a well-structured format, using appropriate headings, subheadings, and bullet points as needed to enhance readability and organization.
    The output should be in text format and not in markdown.
    """

    return prompt_template

def fetch_links(search):
    links = []
    query = quote(search)

    res = requests.get(f"https://tailsx.com/search?q={query}")

    soup = BeautifulSoup(res.content, "html.parser")
    results = soup.find_all('div', class_="results")

    for result in results:
        d = result.find("a", id="link")
        if d != None:
            links.append(d.text)

    return links
