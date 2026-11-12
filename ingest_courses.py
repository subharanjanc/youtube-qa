"""Reproduce the public-caption checkpoint; downloads no video/audio files."""
from pathlib import Path
import json,urllib.request,re,hashlib
import yt_dlp
ROOT=Path(__file__).resolve().parent
COURSES=[('71MW5WeHdz8','RAG'),('AOQyRiwydyo','LangChain'),('DtW_Lc9hYoU','LangGraph')]
START,END=600,2400

def main(topic="RAG"):
    catalogue=json.loads((ROOT/'video_catalogue.json').read_text(encoding='utf-8'))
    video=next(x for x in catalogue if x['topic']==topic)
    with yt_dlp.YoutubeDL({'quiet':True,'skip_download':True,'no_warnings':True,'noplaylist':True}) as client:
        rag=client.extract_info(video['url'],download=False)
    video['ingested_interval']=[START,END]
    tracks=rag.get('subtitles',{}).get('en')
    kind='manual'
    if not tracks:
        tracks=rag.get('automatic_captions',{}).get('en-orig') or rag.get('automatic_captions',{}).get('en')
        kind='automatic'
    if not tracks:raise RuntimeError('No English captions available; choose an authorized transcript source.')
    track=next(x for x in tracks if x['ext']=='json3')
    with urllib.request.urlopen(track['url'],timeout=45) as response:raw=json.load(response)
    segments=[]
    for event in raw.get('events',[]):
        start=event.get('tStartMs',0)/1000
        text=re.sub(r'\s+',' ',''.join(x.get('utf8','') for x in event.get('segs',[]))).strip()
        if text and START<=start<END:
            segments.append({'start':start,'end':min(start+event.get('dDurationMs',0)/1000,END),'text':text})
    if not segments:raise RuntimeError('No captions in selected interval')
    payload={'video_id':rag['id'],'title':rag['title'],'channel':rag['channel'],
        'language':'en','caption_type':kind,'start_seconds':START,'end_seconds':END,'segments':segments}
    chunks=[];i=0
    while i<len(segments):
        j=i;words=0
        while j<len(segments) and (words<220 or j==i):
            words+=len(segments[j]['text'].split());j+=1
        selected=segments[i:j];text=' '.join(s['text'] for s in selected)
        chunks.append({'id':hashlib.sha256((rag['id']+str(selected[0]['start'])+text).encode()).hexdigest(),
            'text':text,'metadata':{'video_id':rag['id'],'title':rag['title'],'trainer':rag['channel'],
            'topic':topic,'start_seconds':selected[0]['start'],'end_seconds':selected[-1]['end'],
            'url':f'https://www.youtube.com/watch?v={rag["id"]}&t={int(selected[0]["start"])}s',
            'language':'en','caption_type':kind}})
        if j==len(segments):break
        back=j;overlap=0
        while back>i+1 and overlap<35:
            back-=1;overlap+=len(segments[back]['text'].split())
        i=back
    for name,value in [('video_catalogue.json',catalogue),(topic.lower()+'_10m_40m_captions.json',payload),
                       (topic.lower()+'_10m_40m_chunks.json',chunks)]:
        (ROOT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Prepared',len(chunks),'chunks; interval 10:00–40:00. Run setup_search.py next.')

if __name__=='__main__':
    import sys
    main(sys.argv[1] if len(sys.argv)>1 else 'RAG')
