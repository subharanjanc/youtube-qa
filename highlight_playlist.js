const highlightPlayer = (() => {
  let clip = null;
  let player = null;
  let generation = 0;
  let apiPromise;
  let readyTimer;

  function loadApi() {
    if (window.YT?.Player) return Promise.resolve(window.YT);
    if (!apiPromise) {
      apiPromise = new Promise((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error('YouTube player timed out')), 12000);
        window.onYouTubeIframeAPIReady = () => {
          clearTimeout(timeout);
          resolve(window.YT);
        };
        const script = document.createElement('script');
        script.src = 'https://www.youtube.com/iframe_api';
        script.onerror = () => {
          clearTimeout(timeout);
          reject(new Error('YouTube player unavailable'));
        };
        document.head.append(script);
      });
    }
    return apiPromise;
  }

  function stopPlayer() {
    generation += 1;
    clearTimeout(readyTimer);
    player?.destroy?.();
    player = null;
    $('player').replaceChildren();
  }

  async function play() {
    if (!clip) return;
    stopPlayer();
    const current = clip;
    const attempt = generation;
    $('status').textContent = `Opening ${current.trainer}’s clip…`;
    const params = new URLSearchParams({
      enablejsapi: '1', origin: location.origin, start: String(current.start),
      end: String(current.end), autoplay: '1', rel: '0',
      cc_load_policy: '1', cc_lang_pref: 'de'
    });
    const frame = document.createElement('iframe');
    frame.title = current.title;
    frame.allow = 'autoplay; encrypted-media; picture-in-picture';
    frame.allowFullscreen = true;
    frame.referrerPolicy = 'strict-origin-when-cross-origin';
    frame.src = `https://www.youtube.com/embed/${current.video_id}?${params}`;
    $('player').replaceChildren(frame);
    $('watch').hidden = false;
    readyTimer = setTimeout(() => {
      if (attempt === generation) {
        $('status').textContent = 'YouTube has not confirmed playback. Press play inside the video or open it on YouTube.';
      }
    }, 8000);
    try {
      const youtube = await loadApi();
      if (attempt !== generation) return;
      player = new youtube.Player(frame, {
        events: {
          onReady: () => {
            if (attempt === generation) {
              clearTimeout(readyTimer);
              $('status').textContent = `Clip ready for ${current.trainer}. Press play inside the video if it is paused.`;
            }
          },
          onStateChange: event => {
            if (attempt !== generation) return;
            if (event.data === youtube.PlayerState.PLAYING) {
              clearTimeout(readyTimer);
              $('status').textContent = `Playing ${current.trainer}’s clip.`;
            }
            if (event.data === youtube.PlayerState.ENDED) $('status').textContent = 'Best match finished.';
          },
          onAutoplayBlocked: () => {
            if (attempt === generation) {
              clearTimeout(readyTimer);
              $('status').textContent = 'Press play inside the video to continue.';
            }
          },
          onError: () => {
            if (attempt === generation) {
              clearTimeout(readyTimer);
              $('status').textContent = 'This clip could not play here. Open it on YouTube.';
            }
          }
        }
      });
    } catch {
      if (attempt !== generation) return;
      clearTimeout(readyTimer);
      $('status').textContent = 'Press play inside the video, or open it on YouTube.';
    }
  }

  function set(item) {
    stopPlayer();
    clip = item;
    $('bestMatch').hidden = $('details').hidden = !clip;
    $('watch').hidden = true;
    if (clip) $('watch').href = clip.watch_url;
    else $('watch').removeAttribute('href');
    $('evidence').textContent = clip
      ? [clip.caption_type, clip.evidence].filter(Boolean).join('\n\n') : '';
  }

  $('playBest').onclick = () => void play();
  return {set};
})();
