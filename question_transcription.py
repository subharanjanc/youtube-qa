"""Transcribe a short question in memory; no recording is saved to disk."""
MIMES={'audio/webm':('question.webm','audio/webm'),
       'audio/flac':('question.flac','audio/flac'),
       'audio/ogg':('question.ogg','audio/ogg'),
       'audio/mp4':('question.mp4','audio/mp4'),
       'audio/wav':('question.wav','audio/wav'),
       'audio/x-wav':('question.wav','audio/wav')}
MAX_AUDIO_BYTES=8_000_000

def transcribe_question(audio,mime,client=None):
    base=mime.split(';',1)[0].strip().lower()
    if base not in MIMES:raise ValueError('Unsupported recording format. Try Chrome or Edge.')
    if not 512<=len(audio)<=MAX_AUDIO_BYTES:raise ValueError('Recording is empty or too large. Please record one short question.')
    if client is None:
        from openai import OpenAI
        client=OpenAI(timeout=45,max_retries=1)
    name,content_type=MIMES[base]
    result=client.audio.transcriptions.create(model='gpt-4o-mini-transcribe',
        file=(name,audio,content_type),language='en',
        prompt='Technical questions about RAG (retrieval augmented generation), vector databases, hybrid index, hybrid search, embeddings, structured output with Pydantic, ROUGE evaluation metrics, LLM-as-a-judge, LangChain, AWS, Azure, Ansh Lamba, Dhaval Patel, Piyush Garg, Edureka Keerti Purswani, Sudarshan Koirala, Krish Naik, and John Savill.')
    text=result.text.strip()
    # On near-silence a speech model can repeat its vocabulary hint back as if it had been said.
    if text.lower().startswith('technical questions about'):text=''
    if not text:raise ValueError('No speech was recognized. Speak closer to the microphone and try again.')
    if len(text)>1000:raise ValueError('Please ask a shorter question (up to 1000 characters).')
    return text
