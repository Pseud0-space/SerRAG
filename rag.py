from SerpLocal.utils import prompt_template, extract_text, summaries_web_groq
from SerpLocal.search import results

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
import logging

from dotenv import load_dotenv

load_dotenv()

logging.disable(logging.CRITICAL)

def get_context_retriever_chain(vector_store):
    llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain

def get_conversational_rag_chain(retriever_chain): 
    
    llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input, vectordb, chat_history):
    retriever_chain = get_context_retriever_chain(vectordb)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history" : chat_history,
        "input": user_input
    })
    
    return response['answer']

def extract_urls(num):
    ret = []

    ext = results(query)["Organic Results"][:num]
    for x in ext:
        ret.append(x["Url"])

    return ret

query = input("\n[WEB SEARCH] >> ")
urls = extract_urls(10)
print(f"\nURLs Fetched successfully | COUNT : {len(urls)}\n")

web_summaries = []

if len(urls) > 0:
    for url in urls:
        content = extract_text(url)

        if content != "skip":
            prompt = prompt_template(content, query)
            
            #summary = summaries_web_ollama(OLLAMA_MODEL, prompt)  # Using ollama for summarization
            summary = summaries_web_groq(prompt)

            web_summaries.append(summary)
            print(f"LINK : {url} |  DONE")

        else:
            print(f"LINK : {url} |  SKIPPED")

    print("Web Summaries Generation Complete.")

    data = "\n\n".join(txt for txt in web_summaries)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap = 1000)
    txt_data = splitter.split_text(data)

    vectordb = Chroma.from_texts(txt_data, embeddings)
    HISTORY = []
    TERM = False

    while TERM == False:
        if TERM == True:
            break 
        
        query_prompt = input("\n[PROMPT] >>> ")

        if query_prompt == "/bye" or query_prompt == "/exit":
            TERM = True 
            break
            
        else:
            response = get_response(query_prompt, vectordb, chat_history=HISTORY)
            HISTORY.append(HumanMessage(content=query_prompt))
            HISTORY.append(AIMessage(content=response))

            print(f"\n\nRESPONSE:\n\n{response}")

else:
    print("\nERROR!! - Empty URL List, Try Again")
