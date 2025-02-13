#!/usr/bin/env python3

"""
hulk_ai.py
"""

import os
import sys
import json

# Load environment variables
from dotenv import load_dotenv
import openai

# Import required langchain modules
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory

# Custom conversational retrieval chain
from langchain.chains import ConversationalRetrievalChain
from typing import Any, Dict


# Configurations
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY not found in .env or environment.")
    sys.exit(1)

openai.api_key = OPENAI_API_KEY

FAQ_FILE = "faq.json"
ASSISTANT_NAME = "HULK AI"

SYSTEM_PROMPT = f"""You are {ASSISTANT_NAME}, an AI assistant trained on the LIV.
If you lack info, respond with "I'm sorry, I don't have information on that."
Do not invent facts beyond what is provided in the documents.
Answer in a concise and helpful manner.
"""

CHUNK_SIZE = 600
CHUNK_OVERLAP = 50
TOP_K = 3


# FAQ Data from LIV site
def load_faq_data(path: str):
    if not os.path.exists(path):
        print(f"FAQ file not found: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print("faq.json must contain a list of {title, content}")
        sys.exit(1)
    return data


# Vector DB setup
def build_vector_store(faq_data):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = []
    for entry in faq_data:
        title = entry.get("title", "")
        content = entry.get("content", "")
        chunks = splitter.split_text(content)
        for chunk in chunks:
            doc = Document(page_content=chunk, metadata={"title": title})
            docs.append(doc)
    
    if not docs:
        print("ERROR: No documents found in FAQ data. Ensure your JSON has valid 'title' and 'content' fields.")
        sys.exit(1)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    faiss_db = FAISS.from_documents(docs, embeddings)
    return faiss_db


# Custom Chain
class CustomConversationalRetrievalChain(ConversationalRetrievalChain):

    def _call(self, inputs: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        # Use parent class processing
        result = super()._call(inputs, **kwargs)

        # Debugging: Ensure retriever returned documents
        if "source_documents" in result:
            if not result["source_documents"]:
                print("WARNING: No relevant documents found. The retriever may not be working correctly.")
        else:
            result["source_documents"] = []  # Avoid KeyError
        
        return result


def create_chain(faiss_db):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_key=OPENAI_API_KEY,
        temperature=0.0
    )
    retriever = faiss_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )


    # For testing, if everything is working fine
    test_query = "What is Liv?"
    test_docs = retriever.get_relevant_documents(test_query)
    if not test_docs:
        print("ERROR: FAISS retrieval is not returning any documents. Check if your embeddings are working.")
        sys.exit(1)

    chain = CustomConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer",
        verbose=False
    )
    return chain



def main():
    # Load FAQ data
    faq_data = load_faq_data(FAQ_FILE)
    # Build vector store
    faiss_db = build_vector_store(faq_data)
    # Create chain
    chain = create_chain(faiss_db)

    print(f"Hi, I am {ASSISTANT_NAME}, How can I assist you today?")
    print("Type 'bye' or 'see you' to end.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["bye", "see you"]:
            print(f"{ASSISTANT_NAME}: Goodbye!")
            break
        if not user_input:
            continue

        # Prepend system prompt to reduce hallucination
        merged_query = f"{SYSTEM_PROMPT}\nUser: {user_input}"

        with get_openai_callback() as cb:
            outputs = chain.invoke({"question": merged_query})

        bot_answer = outputs.get("answer", "I'm sorry, I don't have information on that.")

        print(f"{ASSISTANT_NAME}: {bot_answer}\n")

if __name__ == "__main__":
    main()
