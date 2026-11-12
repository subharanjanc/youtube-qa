"""Add more COMMAND (@cmd_labs) AI / agentic-AI videos (chosen 10 Nov 2026).
The video already indexed ("LLM-as-a-Judge Evals with LangSmith") is kept as it is."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from ingest_dhaval import main

if __name__ == '__main__':
    keep = json.loads((ROOT / 'command_catalogue.json').read_text(encoding='utf-8'))
    keep = [item for item in keep if item['video_id'] == 'qoPYlLg7_rY']
    main([
        ('wFPYjVwgxEk', 'Agents'),      # 9 AI Agent Concepts Every Business Leader Needs
        ('Bda73OPFGss', 'Agents'),      # Agent Tools vs. Agent Skills
        ('ivjZ5XJZ2bs', 'Agents'),      # A ReAct Agent Demo AND An Agentic RAG Demo
        ('QuYlSSYWogE', 'RAG'),         # Master AI Reranking in 23 Minutes
        ('uvJswVmnatM', 'Evaluation'),  # Learn LLM-as-a-Judge in 15 minutes
        ('t9kz9P6cEYM', 'Agents'),      # Computer Use Agents Deep Dive
        ('AesaLzudK9M', 'Agents'),      # Hierarchical Agents vs. Sequential Agents with CrewAI
        ('MDb81aT0V4w', 'RAG'),         # Build a private RAG Assistant with CrewAI, Ollama & AgentOps
        ('upTlhKQbnsE', 'Agents'),      # Production-Grade AI Agents Masterclass
        ('Vq3oCOVVwwM', 'RAG'),         # The Hidden Power of Rerankers in AI Systems
    ], trainer='COMMAND', channel='command', prefix='command', keep=keep)
