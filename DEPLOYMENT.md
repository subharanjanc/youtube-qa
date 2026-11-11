# Deploy the YouTube Q&A app

1. Prepare a private Git repository for the Render service. Include the Dockerfile, .dockerignore, render.yaml, requirements.txt, the app source and browser assets, all caption/chunk/catalogue JSON files, and the two files in models/minilm. Some of those data files are ignored by the local .gitignore, so explicitly include the deployment assets. Keep credentials in the hosting dashboard.
2. On Render, deploy the repository as a Docker web service or use the included Blueprint. Set OPENAI_API_KEY and DEMO_PASSWORD. The container builds its ChromaDB index from cached captions during the image build. Check /healthz, then open the Render URL and sign in as demo with DEMO_PASSWORD.
3. Run python prepare_vercel.py https://YOUR-SERVICE.onrender.com. Deploy the generated vercel-frontend folder as a Vercel project with the Other preset. The rewrite rules route the web page and its API to Render under one browser origin.
4. Open the Vercel URL. Verify the password prompt, a typed question, a spoken question, a follow-up, and Play best match. If a video restricts embedding, use its timestamped YouTube link.

For new caption ingestion, set YOUTUBE_CAPTION_PROXY in the environment that runs the ingestion script. Use the full authenticated proxy URL from the provider, including the scheme, host, and port. URL-encode special characters in the username or password. Caption metadata and caption requests both use the same yt-dlp proxy. Rebuild the image after adding the resulting caption, chunk, and catalogue files.

Run one Render instance for the demo. Conversation memory is held in the Python process and clears when Render restarts it.
