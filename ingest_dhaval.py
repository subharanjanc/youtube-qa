"""Fetch the two approved Codebasics videos' public English captions only."""
from pathlib import Path
import json,re,hashlib,os
from urllib.parse import urlsplit
import yt_dlp
from yt_dlp.networking import Request
ROOT=Path(__file__).resolve().parent
VIDEOS=[('dDkynerzV-Q','RAG'),('vJOGC8QJZJQ','LangChain')]

def caption_options():
    options = {'quiet': True, 'skip_download': True, 'no_warnings': True, 'noplaylist': True}
    proxy = os.environ.get('YOUTUBE_CAPTION_PROXY', '').strip()
    if proxy:
        try:
            url = urlsplit(proxy)
            valid = url.scheme in ('http', 'https', 'socks4', 'socks5', 'socks5h') and url.hostname and url.port
        except ValueError:
            valid = False
        if not valid:
            raise ValueError('YOUTUBE_CAPTION_PROXY must be a proxy URL with a host and port.')
        options['proxy'] = proxy
    return options

def main(videos=None, trainer='Dhaval Patel', channel='codebasics', prefix='dhaval', translate=None, keep=()):
    # keep = catalogue entries already indexed: they stay as they are (no re-download, no duplicate chunks).
    registry=[{**item,'stem':item.get('stem') or prefix+'_'+item['topic'].lower()} for item in keep]
    for vid,topic in videos or VIDEOS:
        with yt_dlp.YoutubeDL(caption_options()) as y:
            info=y.extract_info('https://www.youtube.com/watch?v='+vid,download=False)
        if channel not in info.get('channel','').lower():raise ValueError('Unexpected channel')
        tracks=info.get('subtitles',{}).get('en');kind='manual'
        if not tracks:
            automatic=info.get('automatic_captions',{})
            tracks=automatic.get('en-orig') or automatic.get('en')
            kind='automatic' if automatic.get('en-orig') or info.get('language')=='en' else 'automatic translation'
            if translate and automatic.get(info.get('language','')+'-orig'):
                tracks=automatic[info['language']+'-orig']
                kind='English translation of automatic '+info['language']+' captions'
        if not tracks:raise RuntimeError('No English caption track for '+vid)
        track=next(t for t in tracks if t['ext']=='json3')
        with y.urlopen(Request(track['url'],headers=info.get('http_headers'))) as response:
            raw=json.load(response)
        end=float(info['duration']);segments=[]
        for e in raw.get('events',[]):
            start=e.get('tStartMs',0)/1000
            text=re.sub(r'\s+',' ',''.join(x.get('utf8','') for x in e.get('segs',[]))).strip()
            if text and 0<=start<end:segments.append({'start':start,'end':min(end,start+e.get('dDurationMs',0)/1000),'text':text})
        if not segments:raise ValueError('Empty captions')
        if translate and kind.startswith('English translation of'):
            segments=translate(segments)
        data={'video_id':vid,'title':info['title'],'channel':info['channel'],'trainer':trainer,
            'topic':topic,'language':'en','caption_type':kind,'start_seconds':0,'end_seconds':end,'segments':segments}
        chunks=[];i=0
        while i<len(segments):
            j=i;words=0
            while j<len(segments) and words<220:words+=len(segments[j]['text'].split());j+=1
            group=segments[i:j];text=' '.join(c['text'] for c in group)
            chunks.append({'id':hashlib.sha256((vid+str(group[0]['start'])+text).encode()).hexdigest(),'text':text,
                'metadata':{'video_id':vid,'title':info['title'],'trainer':trainer,'channel':info['channel'],
                    'topic':topic,'start_seconds':group[0]['start'],'end_seconds':group[-1]['end'],
                    'url':f'https://www.youtube.com/watch?v={vid}&t={int(group[0]["start"])}s','language':'en','caption_type':kind}})
            if j==len(segments):break
            back=j;overlap=0
            while back>i+1 and overlap<35:back-=1;overlap+=len(segments[back]['text'].split())
            i=back
        # 2026-11-10 · several videos may share a topic (Krish Naik has 3 on RAG), so the 2nd and 3rd get
        #   _2, _3 in their file names; the first keeps the old name, so earlier trainers are unchanged.
        same_topic=sum(item['topic']==topic for item in registry)
        stem=prefix+'_'+topic.lower()+(f'_{same_topic+1}' if same_topic else '')
        for suffix,value in [('captions',data),('chunks',chunks)]:
            (ROOT/(stem+'_'+suffix+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
        registry.append({**{k:data[k] for k in ('video_id','title','channel','trainer','topic','start_seconds','end_seconds')},'stem':stem})
        print(topic,info['title'],'duration',end,'cues',len(segments),'chunks',len(chunks),flush=True)
    (ROOT/(prefix+'_catalogue.json')).write_text(json.dumps(registry,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
