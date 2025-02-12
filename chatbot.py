# chatbot_demo.ipynb or chatbot_demo.py

import os
import json

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env

openai_api_key = os.getenv("OPENAI_API_KEY")
assert openai_api_key, "Please set your OPENAI_API_KEY in .env"

# 1. Imports from LangChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

# 2. Load your FAQ data from faq.json
with open('faq.json', 'r', encoding='utf-8') as f:
    faq_data = json.load(f)

# Each item in faq_data has {"title": "...", "content": "..."}
# Convert each FAQ entry into a LangChain Document, storing the "title" as metadata
documents = []
for faq in faq_data:
    doc = Document(
        page_content=faq["content"],
        metadata={"title": faq["title"]}
    )
    documents.append(doc)

# 3. Create embeddings and build an in-memory FAISS vector store
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = FAISS.from_documents(documents, embeddings)

# 4. Set up a ConversationalRetrievalChain with memory
#    - This allows multi-turn Q&A with context from previous turns
#    - chain_type="stuff" simply "stuffs" retrieved docs into the prompt
#      (you can experiment with "map_reduce", "refine", etc. if your docs are large)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# ChatOpenAI is a wrapper around the ChatGPT-style models (gpt-3.5-turbo, gpt-4, etc.)
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", 
    openai_api_key=openai_api_key,
    temperature=0.0
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True  # Optionally return the docs used
)

# 5. Interaction loop
print("Welcome to the Fintech Chatbot! (type 'exit' to quit)\n")

while True:
    user_query = input("User: ")
    if user_query.lower().strip() in ["exit", "quit"]:
        print("Exiting chatbot. Goodbye!")
        break
    
    # Use the chain to get an answer
    response = qa_chain({"question": user_query})

    answer = response["answer"]
    source_docs = response["source_documents"]  # If you want to see which docs were used

    print(f"Bot: {answer}")

    # (Optional) If you want to print out which FAQ 'title' was retrieved:
    # for i, doc in enumerate(source_docs, start=1):
    #     print(f"  Source {i}: {doc.metadata.get('title', 'No Title')}")

    # (Optional) Collect user feedback
    feedback = input("Was this answer helpful? (y/n): ")
    if feedback.lower().startswith('n'):
        print("Sorry about that! I'll try to improve.")
        # Here you could log the negative feedback or escalate to a human, etc.
    else:
        print("Great! Glad to help.")
    
    print("---")
