"""Fetch public timestamped captions for Keerti Purswani's tutorials."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest_dhaval import main

if __name__ == '__main__':
    main([
        ('wzssm02u35I', 'RAG'),
        ('bcCFTk_uwUw', 'LangChain'),
        ('wvNgKx2e_LA', 'Agents'),
    ], trainer='Keerti Purswani', channel='keerti purswani', prefix='keerti')
