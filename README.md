# YouTube Q&A: "Highlight in YouTube about the correct answer"

Ask a question about AI, typed or spoken. The app searches the captions of **42 YouTube lessons from 11 trainers**,
writes a short answer from the trainers' own words, and plays the best matching 3–4 minute clip (**Play best match**).

**Live demo:** <https://youtube-qa-ruddy.vercel.app/> (password-protected: user `demo`)

Ironhack AI Engineering bootcamp, Project 4: Multimodal AI ChatBot for YouTube Video QA.

---

## What is inside

| Part | Technology | Files |
|---|---|---|
| Caption store and search | **ChromaDB** (one collection per trainer) + local **MiniLM** embeddings (`all-MiniLM-L6-v2`, ONNX, CPU) + BM25 keyword score; **hybrid search**, merged 60 % meaning / 40 % keywords | `video_qa.py`, `models/minilm/` |
| Best clip and answer | OpenAI **gpt-4o-mini** chooses real caption IDs, so clip times always come from the captions, never from the model | `youtube_highlights.py` |
| Agent | **LangChain** tool-calling agent with three tools: `search_transcript`, `lookup_video` (find courses), `summarize_lesson` | `highlight_agent.py` |
| Conversation memory | Last **6 exchanges** per browser tab, in RAM; follow-ups such as "explain it simpler" are rewritten into full questions | `highlight_memory.py` |
| Speech recognition | Browser **MediaRecorder** → OpenAI **`gpt-4o-mini-transcribe`** (English, vocabulary hint), then clean-up of filler words and mishearings | `question_recorder.js`, `question_transcription.py` |
| Web page | HTML/JS with the YouTube embedded player | `youtube_highlights.html`, `highlight_playlist.js` |
| Evaluation | **LangSmith** experiments with LLM judges | `YouTube_QA_LangSmith_Evaluation.ipynb`, `eval/` |
| Hosting | **Render** (Docker) behind **Vercel** (public address) | `Dockerfile`, `render.yaml`, `prepare_vercel.py`, `DEPLOYMENT.md` |

### Trainers and videos

| Trainer | Videos | Topics |
|---|---:|---|
| Ansh Lamba | 4 | RAG, LangChain, AWS, Azure |
| Dhaval Patel (codebasics) | 2 | RAG, LangChain |
| Piyush Garg | 3 | RAG, LangChain, Agents (Hindi audio, English-translated captions) |
| Edureka | 3 | RAG, LangChain, Agents |
| Keerti Purswani | 3 | RAG, LangChain, Agents |
| Hugging Face | 1 | Evaluation |
| COMMAND (@cmd_labs) | 11 | Agents, RAG (incl. reranking), Evaluation |
| KGP Talkie | 1 | LangChain |
| Sudarshan Koirala | 3 | RAG, Agents, Evaluation |
| Krish Naik | 8 | RAG, LangChain, Agents, Evaluation |
| John Savill | 3 | Agents, Azure |

Captions, chunks (~220 words, ~35-word overlap) and catalogues are the `*_captions.json`, `*_chunks.json` and
`*_catalogue.json` files. The Chroma database is built from them on first start (and during the Docker build).

### How one question flows

1. The question (typed or transcribed) is cleaned: filler words removed, mishearings such as "rack" → RAG fixed.
2. A follow-up is rewritten into a standalone question using the conversation memory.
3. Hybrid search over every trainer's Chroma collection; the 4 best-fitting trainers are compared.
4. gpt-4o-mini picks one 15–60 s explanation by caption ID; the app plays it with context (3 or 4 minutes).
5. If no single passage answers it, a written answer is built from several passages with sources, and the
   best-matching passage still becomes the playable clip.
6. If OpenAI is unreachable, the closest passages are still shown, because search runs locally.

**RAG type:** a traditional (advanced) RAG pipeline with an agentic router. It runs one pass
(retrieve → LLM → answer); there is no ReAct-style loop, so it is not agentic RAG.

---

## Run it locally

```
pip install -r requirements.txt
set OPENAI_API_KEY=...            (Windows; use export on macOS/Linux)
python run_app.py
```

The app opens at <http://127.0.0.1:8901/> (or the next free port). On Windows you can double-click
`Start YouTube QA App.bat`. Create the OpenAI key with expiry **"Never"**; a 30-day key stops working after a month.

---

## Evaluation (LangSmith)

* Notebook: `YouTube_QA_LangSmith_Evaluation.ipynb` (set `RUN_LANGSMITH = True`, `SAMPLE_SIZE = None`).
* Script: `python eval/run_eval.py --prefix <version>`; reports are written to `eval/results/`.
* 15 test questions with reference answers and reference video moments (`eval/dataset.json`); judge: gpt-4.1-mini.
  Needs `OPENAI_API_KEY` and `LANGSMITH_API_KEY`.

| Version | Correctness | Relevance | Groundedness | Retrieval | Timestamp accuracy | Avg. response |
|---|---:|---:|---:|---:|---:|---:|
| 5 Nov, baseline | 87 % | 100 % | 100 % | 100 % | 83 % | 3.3 s |
| 5 Nov, v3-neighbours-b | 100 % | 100 % | 100 % | 100 % | 92 % | 3.0 s |
| **11 Nov, v4-nov11 (11 trainers)** | **100 %** | **100 %** | **100 %** | **100 %** | **92 %** | **3.1 s** |

The one timestamp miss ("How does agentic RAG differ from normal RAG?") had a correct answer, but its clip started at
07:48 instead of the reference 03:58–05:34.

---

## Deployment

See `DEPLOYMENT.md`. Render builds the `Dockerfile` (it pre-builds the Chroma index) and runs `serve_render.py`;
set `OPENAI_API_KEY` and `DEMO_PASSWORD` in Render → Environment. Vercel forwards the public address to Render
(`prepare_vercel.py` generates its rewrite rules). `.github/workflows/keep-awake.yml` pings `/healthz` every 10 minutes so
the free Render instance does not sleep. The page also reconnects by itself if the server restarts.

## Adding videos

`ingest_*.py` download public English captions and write the caption, chunk and catalogue files
(for example `python ingest_krish.py`; `ingest_command.py` shows how to add videos to an existing trainer).
Then add the trainer to `highlight_trainers.py`, the prefix list in `youtube_highlights.py` and the dropdown in
`youtube_highlights.html`, and restart.

## Known limits

* The agent makes one tool choice per question (no multi-step loop).
* Memory is in RAM, so a restart clears it.
* Some videos block embedding; the page then offers "Open at this time on YouTube".
