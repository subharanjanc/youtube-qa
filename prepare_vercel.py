"""Generate the Vercel project after the Render service has a URL."""
import argparse
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('render_url', help='https://YOUR-SERVICE.onrender.com')
    args = parser.parse_args()
    origin = args.render_url.rstrip('/')
    if not re.fullmatch(r'https://[a-z0-9-]+\.onrender\.com', origin):
        parser.error('Enter the HTTPS origin shown in Render, without a path.')
    folder = Path(__file__).resolve().parent / 'vercel-frontend'
    folder.mkdir(exist_ok=True)
    paths = ('/', '/question_recorder.js', '/highlight_playlist.js',
             '/find', '/transcribe', '/history', '/reset', '/healthz')
    config = {'rewrites': [{'source': path, 'destination': origin + path} for path in paths]}
    (folder / 'vercel.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    print(folder)


if __name__ == '__main__':
    main()
