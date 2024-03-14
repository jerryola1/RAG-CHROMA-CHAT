from rich import print
from langchain.docstore.document import Document
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import requests
from lxml import etree
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import logging
import time
import json
import re


# RAG
def rag(chunks, collection_name):
    vectorstore = Chroma.from_documents(
        documents=documents,
        collection_name=collection_name,
        embedding=embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text'),
    )
    retriever = vectorstore.as_retriever()

    prompt_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | local_llm
        | StrOutputParser()
    )
    result = chain.invoke("What is the use of Text Splitting?")
    print(result)

#     # 2. Recursive Character Text Splitting
# print("#### Recursive Character Text Splitting ####")

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# with open('data.txt', 'r', encoding='utf-8') as file:
#     text = file.read()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size = 650, chunk_overlap=65) # ["\n\n", "\n", " ", ""] 65,450
# print(text_splitter.create_documents([text])) 


###  START TEXT EXTRACTION ================================================================================================

# Configure logging
#logging.basicConfig(filename='extraction.log', level=logging.ERROR)

#def extract_text_from_url(url, max_retries=3, timeout=10):
    retries = 0
    while retries < max_retries:
        try:
            print(f"Extracting text from URL: {url}")
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style tags
            for script in soup(['script', 'style']):
                script.decompose()
            
            # Extract text from the page
            text = soup.get_text(separator='\n', strip=True)
            
            return text
        except (requests.exceptions.RequestException, ValueError) as e:
            logging.error(f"Error fetching URL: {url}")
            logging.error(f"Error message: {str(e)}")
            retries += 1
            time.sleep(1)  # Delay before retrying
    
    return None

#def extract_text_from_sitemap(sitemap_url, output_file, max_retries=3, timeout=10):
    retries = 0
    while retries < max_retries:
        try:
            print(f"Extracting text from sitemap: {sitemap_url}")
            response = requests.get(sitemap_url, timeout=timeout)
            response.raise_for_status()
            sitemap_content = response.content
            
            root = etree.fromstring(sitemap_content)
            
            namespaces = {'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            
            extracted_data = {}
            problematic_urls = []
            
            start_time = time.time()
            
            for loc_element in root.xpath('//sitemap:url/sitemap:loc', namespaces=namespaces):
                url = loc_element.text
                
                # Exclude unwanted pages based on their URL patterns
                excluded_patterns = [
                    '/contact',
                    '/terms-and-conditions',
                    '/privacy-policy',
                    '/search',
                    '/events',
                    '/news',
                ]
                
                if any(pattern in url for pattern in excluded_patterns):
                    continue
                
                text = extract_text_from_url(url, max_retries, timeout)
                if text:
                    extracted_data[url] = text
                else:
                    problematic_urls.append(url)
            
            end_time = time.time()
            extraction_time = end_time - start_time
            
            # Save extracted data to a file in dictionary format
            with open(output_file, 'w', encoding='utf-8') as file:
                json.dump(extracted_data, file, ensure_ascii=False, indent=4)
            
            print(f"Extracted {len(extracted_data)} pages from the sitemap.")
            print(f"Encountered {len(problematic_urls)} problematic URLs.")
            print(f"Extraction time: {extraction_time:.2f} seconds.")
            
            # Save problematic URLs to a separate file
            with open('problematic_urls.txt', 'w') as file:
                file.write('\n'.join(problematic_urls))
            
            return len(extracted_data)
        except (requests.exceptions.RequestException, ValueError) as e:
            logging.error(f"Error fetching sitemap: {sitemap_url}")
            logging.error(f"Error message: {str(e)}")
            retries += 1
            time.sleep(1)  # Delay before retrying
    
    return 0

# Example usage
#sitemap_url = 'https://www.hull.ac.uk/sitemap.xml'
#output_file = 'extracted_text.txt'
#extracted_count = extract_text_from_sitemap(sitemap_url, output_file)

#print(f"Total pages extracted: {extracted_count}")

###  END TEXT EXTRACTION ================================================================================================


###############################################################################################################

### START TEXT PREPROCESSING ================================================================================================

# def preprocess_text(text):
#     # Remove the page title
#     text = text.split("\\n", 1)[1] if "\\n" in text else text

#     # Extract and replace useful URLs with placeholders
#     url_pattern = r'(https?://[^\s]+)'
#     urls = re.findall(url_pattern, text)
#     for i, url in enumerate(urls, start=1):
#         text = text.replace(url, f'<URL_{i}>')

#     # Remove repetitive elements and unwanted patterns
#     patterns_to_remove = [
#         r"\\nSkip to\\ncontent\\nor\\nfooter\\n",
#         r"Jump to section\.\.\.",
#         r"Browser does not support script\.",
#         r"©\nUniversity of Hull, \d{4}"
#     ]
#     for pattern in patterns_to_remove:
#         text = re.sub(pattern, "", text, flags=re.IGNORECASE)

#     # Remove page title
#     page_title_pattern = r"^.*\| University of Hull\\n"
#     text = re.sub(page_title_pattern, "", text)

#     # Handle newline characters
#     text = re.sub(r"\\n", " ", text)

#     # Normalize and clean the text
#     text = text.lower()
#     text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
#     text = re.sub(r"\s+", " ", text)  # Replace multiple whitespace characters with a single space

#     # Replace placeholders with original URLs
#     for i, url in enumerate(urls, start=1):
#         text = text.replace(f'<URL_{i}>', url)

#     return text.strip()

# def chunk_text(text, chunk_size=200):
#     words = text.split()
#     chunks = []

#     for i in range(0, len(words), chunk_size):
#         chunk = " ".join(words[i:i+chunk_size])
#         chunks.append(chunk)

#     return chunks

# def process_extracted_data(extracted_data, output_file):
#     preprocessed_data = []
#     chunk_counter = {}

#     for url, text in extracted_data.items():
#         preprocessed_text = preprocess_text(text)
#         chunks = chunk_text(preprocessed_text)

#         if url not in chunk_counter:
#             chunk_counter[url] = 1

#         for chunk in chunks:
#             chunk_data = {
#                 "url": url,
#                 "chunk_id": f"{url}_{chunk_counter[url]}",
#                 "text": chunk
#             }
#             preprocessed_data.append(chunk_data)
#             chunk_counter[url] += 1

#     with open(output_file, "w", encoding="utf-8") as file:
#         json.dump(preprocessed_data, file, ensure_ascii=False, indent=4)

# # Read the extracted data from the file
# with open("extracted_text.txt", "r", encoding="utf-8") as file:
#     extracted_data = json.load(file)

# output_file = "preprocessed_data1.txt"
# process_extracted_data(extracted_data, output_file)

### END TEXT PREPROCESSING ================================================================================================

###############################################################################################################


# from langchain_community.document_loaders import AsyncChromiumLoader
# from langchain_community.document_transformers import BeautifulSoupTransformer

# # Load HTML
# loader = AsyncChromiumLoader(["https://www.beyond-events.co.uk/"])
# html = loader.load()

# # Transform
# bs_transformer = BeautifulSoupTransformer()
# docs_transformed = bs_transformer.transform_documents(html, tags_to_extract=["span"])

# # Result
# print(docs_transformed[0].page_content[0:500])


from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_extraction_chain
from langchain_openai import ChatOpenAI

# URLs to scrape (replace with the desired pages from the University of Hull website)
urls = [
    "https://www.hull.ac.uk/study/webinars.aspx",
    # Add more URLs as needed
]

# Load HTML using AsyncChromiumLoader
# loader = AsyncChromiumLoader(urls)
# docs = loader.load()

# # Transform HTML using BeautifulSoupTransformer
# bs_transformer = BeautifulSoupTransformer()
# docs_transformed = bs_transformer.transform_documents(docs, tags_to_extract=["p", "text-h2", "h2", "h3", "span", "text-h1"])

# # Split the text into chunks
# splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
# splits = splitter.split_documents(docs_transformed)

# # Define schema for extraction
# schema = {
#     "properties": {
#         "title": {"type": "string"},
#         "content": {"type": "string"},
#     },
#     "required": ["title", "content"],
# }

# Initialize LLM
# llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

# Create extraction chain
# extraction_chain = create_extraction_chain(schema=schema, llm=llm)

# Extract relevant information from the scraped data
# extracted_data = []
# for split in splits:
#     # extracted = extraction_chain.invoke(split.page_content)
#     extracted_data.append(extracted)

# # Print the extracted data
# for data in extracted_data:
#     print("Title:", data.get("title"))
#     print("Content:", data.get("content"))
#     print("---")

from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer

loader = AsyncHtmlLoader(urls)
docs = loader.load()

html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(docs)
print(docs_transformed[0].page_content)