"""Start the browser app directly, without Jupyter.

Usage: python run_app.py [--no-browser] [port]
Uses port 8901, or the next free port if something else already holds it.
The OpenAI key comes from the OPENAI_API_KEY Windows environment variable.
"""
import os
import socket
import sys
import threading
import webbrowser
from pathlib import Path

# This PC's embeddable Python does not search the script's own folder automatically.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from youtube_highlights import HighlightFinder, start_app


def free_port(first):
    for port in range(first, first + 20):
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1', port)) != 0:
                return port
    raise SystemExit(f'No free port found between {first} and {first + 19}.')


def main():
    args = [a for a in sys.argv[1:] if a != '--no-browser']
    if not os.environ.get('OPENAI_API_KEY'):
        print('WARNING: OPENAI_API_KEY is not set. Voice and written answers will not work;')
        print('caption search will still show the closest passages.')
    print('Loading the search model and caption collections...')
    finder = HighlightFinder()
    server, url = start_app(finder, port=free_port(int(args[0]) if args else 8901))
    print('App is running at', url)
    print('Keep this window open while you use it. Close it to stop the app.')
    if '--no-browser' not in sys.argv:
        webbrowser.open(url)
    threading.Event().wait()


if __name__ == '__main__':
    main()
