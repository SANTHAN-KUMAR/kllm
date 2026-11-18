// Server-sent events with reconnect. EventSource retries on its own and sends Last-Event-ID when the stream
// carried ids, so a session timeline resumes where it stopped. onState(open: bool) drives the freshness note.

export function stream(url, { onEvent, onState, named = [] } = {}) {
  let es = null, closed = false, retry = 1000, timer = null;
  const open = () => {
    if (closed) return;
    es = new EventSource(url);
    es.onopen = () => { retry = 1000; onState && onState(true); };
    const handle = ev => {
      let data = null;
      try { data = JSON.parse(ev.data); } catch { return; }
      onEvent && onEvent(data, ev);
    };
    es.onmessage = handle;
    named.forEach(n => es.addEventListener(n, handle));
    es.onerror = () => {
      onState && onState(false);
      if (es.readyState === EventSource.CLOSED) {      // the server refused: back off and reopen
        es.close();
        timer = setTimeout(open, retry);
        retry = Math.min(retry * 2, 30000);
      }
    };
  };
  open();
  return () => { closed = true; clearTimeout(timer); if (es) es.close(); };
}
