// Record on the device, then transcribe only after Stop. Avoid the browser's
// SpeechRecognition service, which reports a network error in embedded browsers.
const voiceAvailable=!!(navigator.mediaDevices?.getUserMedia&&window.MediaRecorder);
let questionRecorder=null, questionStream=null, questionChunks=[], voiceLimit=null;
let voiceContext=null, voiceAnalyser=null, voiceFrame=null, voicePeak=0, voiceTicks=0;
let leavingPage=false, transcription=null;
const voiceIdle=()=>{
  $('speak').disabled=!voiceAvailable;
  $('stop').disabled=true;
  $('stop').textContent='Stop listening';
  $('find').disabled=false;
  $('reset').disabled=false;
};
function releaseMicrophone(){
  clearTimeout(voiceLimit);clearInterval(voiceFrame);
  questionStream?.getTracks().forEach(track=>track.stop());questionStream=null;
  if(voiceContext){voiceContext.close().catch(()=>{});voiceContext=null;}
  $('level').value=0;
}
if(!voiceAvailable){$('speak').disabled=true;$('status').textContent='Microphone recording is unavailable here. Open this page in Chrome or Edge, or type your question.';}
$('speak').onclick=async()=>{
  $('speak').disabled=true;$('find').disabled=true;$('reset').disabled=true;
  $('status').textContent='Allow microphone access if prompted…';
  try{
    questionStream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true}});
    if(leavingPage){releaseMicrophone();return;}
    const mime=['audio/webm;codecs=opus','audio/webm','audio/ogg;codecs=opus','audio/mp4'].find(type=>MediaRecorder.isTypeSupported(type));
    if(!mime)throw Error('This browser cannot record a supported audio format. Try Chrome or Edge.');
    questionRecorder=new MediaRecorder(questionStream,{mimeType:mime});questionChunks=[];voicePeak=0;voiceTicks=0;
    const AudioCtx=window.AudioContext||window.webkitAudioContext;
    if(AudioCtx){voiceContext=new AudioCtx();await voiceContext.resume();voiceAnalyser=voiceContext.createAnalyser();voiceAnalyser.fftSize=1024;
      voiceContext.createMediaStreamSource(questionStream).connect(voiceAnalyser);
      const samples=new Uint8Array(voiceAnalyser.fftSize);
      // A timer, not requestAnimationFrame: animation frames stop while the tab is in the background,
      // which made a real recording look silent if you switched windows while speaking.
      const meter=()=>{voiceAnalyser.getByteTimeDomainData(samples);let peak=0;for(const value of samples)peak=Math.max(peak,Math.abs(value-128)/128);voicePeak=Math.max(voicePeak,peak);voiceTicks++;$('level').value=Math.min(1,peak*3);};meter();voiceFrame=setInterval(meter,50);}
    const recorder=questionRecorder;
    // 2026-11-11 · FIX: the previous question stayed in the box while you spoke, so if transcription failed
    //   or was slow, "Find explanation" answered the OLD question. Now the box empties when recording
    //   starts; the old text only comes back if nothing was recorded.
    const previousText=$('question').value;$('question').value='';
    const restorePrevious=()=>{if(!$('question').value.trim())$('question').value=previousText;};
    recorder.ondataavailable=e=>{if(questionRecorder===recorder&&e.data.size)questionChunks.push(e.data);};
    recorder.onerror=()=>{
      if(questionRecorder!==recorder) return;
      questionRecorder=null;questionChunks=[];restorePrevious();
      releaseMicrophone();voiceIdle();
      $('status').textContent='Recording failed. Check the microphone or try Chrome/Edge.';
    };
    recorder.onstop=async()=>{
      if(questionRecorder!==recorder) return;
      questionRecorder=null;
      const canMeasure=voiceContext?.state==='running'&&voiceTicks>1;releaseMicrophone();$('stop').disabled=true;
      const audio=new Blob(questionChunks,{type:recorder.mimeType});questionChunks=[];
      if(leavingPage)return;
      if(audio.size<512||(canMeasure&&voicePeak<0.004)){restorePrevious();voiceIdle();$('status').textContent='No microphone sound detected. Check the selected microphone, then try again. Your previous text is kept.';return;}
      $('status').textContent='Transcribing your recording… your words will appear here shortly.';
      const controller=new AbortController();
      const draft=$('question').value;
      transcription=controller;
      // 2026-11-09 · FIX: Stop instantly turned into "Cancel transcription" under the mouse and Find stayed
      //   active, so a double-click on Stop, or clicking Find straight away, silently threw the words away.
      //   Now both stay locked during a normal transcription and only unlock if it is unusually slow.
      $('stop').textContent='Transcribing…';
      $('stop').disabled=true;
      $('find').disabled=true;
      $('reset').disabled=true;
      // The transcription service can retry once, so allow time for both attempts.
      const timeout=setTimeout(()=>controller.abort(),120000);
      const slowNotice=setTimeout(()=>{
        $('status').textContent='Still transcribing. You can wait, cancel, or type and submit a question.';
        $('stop').textContent='Cancel transcription';
        $('stop').disabled=false;
        $('find').disabled=false;
        $('reset').disabled=false;
      },15000);
      try{const response=await apiFetch('/transcribe',{method:'POST',headers:{'Content-Type':audio.type,'X-Highlight-Token':'__TOKEN__'},body:audio,signal:controller.signal});
        const data=await response.json();if(!response.ok)throw Error(data.error||'Transcription failed.');
        if(transcription!==controller) return;
        if(typeof data.text!=='string'||!data.text.trim()) throw Error('No speech was recognized. Try again or type your question.');
        // If you typed while waiting, keep your typing and add the spoken words after it.
        const edited=$('question').value;
        $('question').value=edited!==draft&&edited.trim()?edited.trimEnd()+' '+data.text:data.text;
        $('question').focus();
        $('status').textContent='Voice question captured. Review it above, then click Find explanation.';
      }catch(error){
        // The box stays empty on purpose, so the previous question cannot be submitted by mistake.
        if(transcription===controller) $('status').textContent=(error.name==='AbortError'?'Transcription timed out.':error.message)+' Your spoken question was not captured: speak again or type it.';
      }finally{
        clearTimeout(timeout);clearTimeout(slowNotice);
        if(transcription===controller){transcription=null;voiceIdle();}
      }
    };
    questionRecorder.start();$('stop').disabled=false;
    $('status').textContent='Recording… say your full question, then click Stop listening. Text appears AFTER you stop. The microphone meter should move.';
    voiceLimit=setTimeout(()=>{if(recorder.state==='recording')recorder.stop();},90000);
  }catch(error){releaseMicrophone();voiceIdle();$('status').textContent=error.name==='NotAllowedError'?'Microphone permission was denied. Allow it for this page, then click Speak question again.':error.name==='NotFoundError'?'No microphone was found. Connect/select a microphone or type your question.':error.message;}
};
function cancelTranscription(){
  if(!transcription) return;
  transcription.abort();
  transcription=null;
  voiceIdle();
  $('status').textContent='Transcription canceled. You can type your question or record again.';
}
$('stop').onclick=()=>{
  if(transcription) cancelTranscription();
  else if(questionRecorder?.state==='recording') questionRecorder.stop();
};
window.addEventListener('pagehide',()=>{leavingPage=true;cancelTranscription();if(questionRecorder?.state==='recording')questionRecorder.stop();releaseMicrophone();});
