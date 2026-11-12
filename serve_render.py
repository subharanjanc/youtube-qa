"""Start the hosted YouTube Q&A backend."""
import os
from threading import Event

from youtube_highlights import HighlightFinder, start_app


if __name__ == '__main__':
    password = os.environ.get('DEMO_PASSWORD')
    if not password or not os.environ.get('OPENAI_API_KEY'):
        raise SystemExit('Set DEMO_PASSWORD and OPENAI_API_KEY in Render.')
    server, _ = start_app(HighlightFinder(), port=int(os.environ.get('PORT', '10000')),
                          host='0.0.0.0', password=password)
    print(f'YouTube Q&A ready on port {server.server_port}', flush=True)
    Event().wait()
