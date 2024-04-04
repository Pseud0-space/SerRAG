# Search RAG
Search engine result scraping and RAG-Based Response Generation in Python using Groq API

## Key Features:
* Google search Scraping: Effecient scraping URLs from Google search results, enabling you to gather relevant data for your projects.
* URL Conent Scraping: Effortlessly scrape content from the extracted URLs, saving you time and effort in collecting information from various sources.
* Retrieval Augmented Generation (RAG): Utilize state-of-the-art RAG models for response generation, enhancing the quality and relevance of generated responses.
* Mixtral LLM Integration: Seamlessly integration of  the super-fast Groq-based Mixtral LLM for rapid and efficient response generation, ensuring quick turnaround times for your projects.


## Getting Started

To get started with this project, follow these steps:

*  Clone the repository: ```git clone https://github.com/Pseud0-space/SerRAG.git```
*  Install Dependencies:
      * ```pip install -r requirements.txt```
      * ```playwright install chromium```
 
* Update environment variables ```.env```    
     ```GROQ_API_KEY=<YOUR_GROQ_API_KEY>``` : https://console.groq.com/keys   
     ```GOOGLE_API_KEY=<YOUR_GOOGLE_API_KEY>``` : https://aistudio.google.com/app/apikey
  
*  Run: ```python rag.py```

## Dependencies
Arch Linux users might face issues with python greenlet, to resolve the issue run command   
```sudo pacman -S python-greenlet```

### Note
I am using google embedding-001 model for the embeddings.
