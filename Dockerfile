FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN pip install chromadb==1.5.9 onnxruntime==1.31.0 tokenizers numpy 'langchain-openai>=1.7,<2' 'pydantic>=2.13,<3' yt-dlp
COPY app.zip /tmp/youtube-qa.zip
RUN python -m zipfile -e /tmp/youtube-qa.zip /app
RUN python -c "from video_qa import VideoQA; from youtube_highlights import HighlightFinder; qa=VideoQA(); qa.index(); HighlightFinder(qa)"
CMD ["python", "serve_render.py"]
