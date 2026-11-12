"""Fetch timestamped public English captions for the selected Krish Naik videos (chosen 10 Nov 2026)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest_dhaval import main

if __name__ == '__main__':
    main([
        ('fZM3oX4xEyg', 'RAG'),         # Introduction To Understanding RAG
        ('CK0ExcCWDP4', 'RAG'),         # Hybrid Search RAG With Langchain And Pinecone Vector DB
        ('6SO-8FcSkz4', 'RAG'),         # Prompt Engineering Vs RAG Vs Finetuning Explained Easily
        ('vmy3HgaKJsY', 'LangChain'),   # Most Popular Framework - Langchain vs LangGraph
        ('HodCjnGv8Ag', 'Agents'),      # What Is Agentic RAG?
        ('p4pHsuEf4Ms', 'Agents'),      # Generative AI Vs Agentic AI Vs AI Agents
        ('3Gcm27l-uyQ', 'Evaluation'),  # Langsmith LLMOPS Platform - Debug, Monitor ...
        ('r0q1n8BJ0QI', 'Evaluation'),  # What Is LLM Hallucination And How to Reduce It?
    ], trainer='Krish Naik', channel='krish naik', prefix='krish')
