"""Fetch timestamped public captions for the selected Edureka videos."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest_dhaval import main

if __name__ == '__main__':
    main([
        ('iLO6yEnIuaA', 'RAG'),
        ('V6ddMAUCOys', 'LangChain'),
        ('KEeolG9JlC4', 'Agents'),
    ], trainer='Edureka', channel='edureka', prefix='edureka')
