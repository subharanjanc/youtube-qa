FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY requirements.txt .
RUN pip install chromadb==1.5.9 onnxruntime==1.31.0 tokenizers numpy 'langchain-openai>=1.7,<2' 'pydantic>=2.13,<3' yt-dlp
COPY youtube_highlights.py video_qa.py highlight_trainers.py highlight_memory.py highlight_agent.py question_transcription.py serve_render.py ingest_dhaval.py ./
COPY youtube_highlights.html highlight_playlist.js question_recorder.js ./
COPY *_catalogue.json *_captions.json *_chunks.json ./
COPY models/minilm/model_quint8_avx2.onnx models/minilm/tokenizer.json ./models/minilm/
RUN python -c "from video_qa import VideoQA; from youtube_highlights import HighlightFinder; qa=VideoQA(); qa.index(); HighlightFinder(qa)"
CMD ["python", "serve_render.py"]
