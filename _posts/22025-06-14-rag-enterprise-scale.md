---
layout: post
title: "From Demo to Deployment: Building a Real RAG System at Scale in Industry"
description: My experience in RAG in production in big company scale.
date: 2025-06-15 07:00:00 +0000
categories: [RAG, AI, Langchain]
tags: [RAG, AI, Langchain, Confluence, GPT, Azure, Langfuse]
math: true  # Enable math equations rendering
author: Paul Moise
by: Paul Moise
---
![RAG in production Grade](/assets/img/favicons/2025-06-16-rag-enterprise/thumbnail.png)

# From Demo to Deployment: Building a Real RAG System at Scale in Industry

*Why Retrieval-Augmented Generation is not just a toy—and what no one tells you about scaling it in a company like Amadeus.*

---

## 1. Introduction

Retrieval-Augmented Generation (RAG) has become a buzzword in the LLM ecosystem. Tutorials make it look simple: chunk some documents, add them to a vector store, query with a prompt, and voilà—instant AI-powered search.

But deploying RAG in a real company, with real data, real security policies, and real users… is a different game.

In this post, I share lessons from building and deploying a RAG-based internal chatbot at Amadeus, a global travel tech company. The bot was designed to answer questions over **more than 1,200 Confluence pages of functional specifications**, used by developers, testers, and business analysts.

---

## 2. The Setup

We needed to:

* **Extract, chunk, and index** large volumes of documentation from Confluence.
* Provide **secure, fast** retrieval over hundreds of thousands of tokens.
* Use **GPT-4o via Azure OpenAI** for generation.
* Ensure **traceability** and **observability** for debugging and compliance.

**Stack:**

* Document source: Confluence API (with rate limits + permissions)
* Embedding: `text-embedding-ada-002` via Azure
* Vector store: Azure AI Search (custom index + filters)
* UI: First version in Streamlit, later replaced with React + FastAPI
* Logging: Langfuse + PostgreSQL

### Sample Ingestion Code (LangChain)

```python
from langchain.document_loaders import ConfluenceLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import AzureSearch

# Load and split documents
loader = ConfluenceLoader(base_url="https://confluence.mycompany.com", username="user", api_key="token")
docs = loader.load(space="PRODUCT")
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
split_docs = splitter.split_documents(docs)

# Embed and store in Azure
embedding_model = OpenAIEmbeddings(deployment="ada-002")
vector_store = AzureSearch(index_name="amadeus-docs-index")
vector_store.add_documents(split_docs, embedding=embedding_model)
```

---

## 3. What Tutorials Don’t Tell You

### a. Document Explosion

Chunking 1,200 pages quickly balloons into **tens of thousands of embeddings**. Simple cosine search isn’t enough. You need:

* Good chunking (respecting semantic units)
* Metadata filtering (per team, project, etc.)
* Reranking (BM25, hybrid search) to improve top-k results

### b. Latency

Azure OpenAI and AI Search are powerful, but **latency grows fast** with long prompts and large index sizes. To manage it:

* Used **query caching** for repeated prompts
* Added **rate limiting** and streaming generation

### c. Security & Compliance

You can’t just “upload everything and ask questions.”

* Confluence data must be **access-controlled**: users should only query what they’re allowed to see.
* Azure logs must be scrubbed of **PII or sensitive business logic**.
* All user queries and LLM responses had to be **traceable and stored** (Langfuse + audit DB)

We also faced **security review bottlenecks**: cloud APIs, data storage, even the embedding model had to be reviewed by internal security teams.

---

## 4. What I Learned

* **LLMs hallucinate more when your retrieval is weak.** The better the retrieval (including context filtering and reranking), the less the LLM needs to "guess".
* **Testing RAG systems is hard**: how do you measure "correctness" over natural language answers? We relied on live testing + user feedback.
* **Design for observability early.** Tools like Langfuse were essential for tracing bad answers and debugging pipeline issues.
* **Start small** and modular. We rewrote the entire system from Streamlit → FastAPI backend + React frontend for better scaling and maintainability.

### Sample API Endpoint for Answer Generation

```python
from fastapi import FastAPI, Request
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query_docs(req: QueryRequest):
    context = retriever.get_relevant_documents(req.question)
    response = llm.generate_answer(req.question, context)
    return {"answer": response}
```

---

## 5. What’s Next

There’s still more to improve:

* Automatic **document update detection**
* Smarter **user personalization**
* Using **agent-like memory** to track context across multiple queries

I’m also exploring adding **multi-source retrieval** (e.g., GitHub issues, ServiceNow tickets) to extend the bot’s capabilities.

---

## 6. Bonus: Lessons from the First Implementation

Let’s be honest—the first implementation of a RAG system you find online almost never works in a real company setup.

**Framework choice** is already a dilemma: LangChain? LlamaIndex? Semantic Kernel? CrewAI? Each one has strengths but comes with a learning curve.

For my use case, I chose **LangChain**. In the beginning, it was overwhelming. The abstraction was so high-level and tightly coupled that customizing behavior became painful.

Thankfully, the framework has improved drastically. Today, it allows more modularity and flexibility, but it took trial and error to get there.

Also, while tutorials recommend monitoring tools, implementing something like Langfuse inside a large company is not trivial. It required **long administrative processes** for approvals, data handling reviews, and integration validations.

---

## Conclusion

RAG is a promising architecture—but it’s not plug-and-play in enterprise settings. You need to think about:

* **Scalability**
* **Security**
* **Latency**
* **Traceability**
* **User access rights**

My experience deploying a RAG chatbot at Amadeus taught me more about production AI than any tutorial ever could. I hope this article helps others building LLM applications that actually work in the real world.
