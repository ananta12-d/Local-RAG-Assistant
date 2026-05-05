/* ============================================================
   Local RAG Assistant — Core Application Logic
   ============================================================ */

/* ── CONFIGURATION ── */
const OLLAMA_BASE   = 'http://localhost:11434';
const CHUNK_SIZE    = 500;
const CHUNK_OVERLAP = 80;
const EMBED_MODEL   = 'nomic-embed-text';

/* ── STATE ── */
let vectorDB    = []; // { docId, docName, chunk, embedding, chunkIdx }
let docs        = []; // { id, name, type, status, chunks, progress, error }
let chatHistory = []; // last 20 messages for multi-turn context

/* ── PDF.JS WORKER ── */
pdfjsLib.GlobalWorkerOptions.workerSrc =
  'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

/* ================================================================
   OLLAMA CONNECTION CHECK
================================================================ */
async function checkOllama() {
  try {
    const r = await fetch(`${OLLAMA_BASE}/api/tags`, { signal: AbortSignal.timeout(3000) });
    if (r.ok) {
      const d = await r.json();
      setStatus(true);
      populateModels(d.models || []);
    } else { setStatus(false); }
  } catch { setStatus(false); }
}

function setStatus(online) {
  document.getElementById('statusDot').className    = 'status-dot' + (online ? ' ok' : '');
  document.getElementById('statusLabel').textContent = online ? 'Ollama Connected' : 'Ollama Offline';
}

function populateModels(models) {
  if (!models.length) return;
  const sel = document.getElementById('modelSelect');
  sel.innerHTML = '';
  models.forEach(m => {
    const o = document.createElement('option');
    o.value = m.name; o.textContent = m.name;
    sel.appendChild(o);
  });
  updateStats();
}

checkOllama();
setInterval(checkOllama, 10000);

/* ================================================================
   EMBEDDING & VECTOR SEARCH
================================================================ */
async function embed(text) {
  const r = await fetch(`${OLLAMA_BASE}/api/embeddings`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: EMBED_MODEL, prompt: text })
  });
  if (!r.ok) throw new Error('Embedding failed. Ensure nomic-embed-text is pulled in Ollama.');
  return (await r.json()).embedding;
}

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-10);
}

async function vectorSearch(query, k) {
  const qEmb = await embed(query);
  return vectorDB
    .map(c => ({ ...c, score: cosine(qEmb, c.embedding) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
}

/* ================================================================
   TEXT CHUNKING
================================================================ */
function chunkText(text, size = CHUNK_SIZE, overlap = CHUNK_OVERLAP) {
  const words  = text.split(/\s+/).filter(Boolean);
  const chunks = [];
  for (let i = 0; i < words.length; i += size - overlap) {
    const chunk = words.slice(i, i + size).join(' ');
    if (chunk.trim().length > 20) chunks.push(chunk);
    if (i + size >= words.length) break;
  }
  return chunks;
}

/* ================================================================
   DOCUMENT PARSERS
================================================================ */
async function parsePDF(file) {
  const buf = await file.arrayBuffer();
  const pdf = await pdfjsLib.getDocument({ data: buf }).promise;
  let text = '';
  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const tc   = await page.getTextContent();
    text += tc.items.map(s => s.str).join(' ') + '\n';
  }
  return text;
}

async function parseTXT(file)  { return await file.text(); }

async function parseDOCX(file) {
  const buf = await file.arrayBuffer();
  return (await mammoth.extractRawText({ arrayBuffer: buf })).value;
}

async function parseImage(file) {
  return new Promise((resolve, reject) => {
    Tesseract.recognize(file, 'eng', { logger: () => {} })
      .then(r => resolve(r.data.text))
      .catch(reject);
  });
}

/* ================================================================
   FILE PROCESSING PIPELINE
================================================================ */
function getType(name) {
  const ext = name.split('.').pop().toLowerCase();
  if (ext === 'pdf') return 'pdf';
  if (['png','jpg','jpeg','webp','bmp'].includes(ext)) return 'img';
  if (['doc','docx'].includes(ext)) return 'doc';
  return 'txt';
}

async function processFile(file) {
  const id   = Date.now() + Math.random();
  const type = getType(file.name);
  const doc  = { id, name: file.name, type, status: 'processing', chunks: 0, progress: 0 };
  docs.push(doc);
  renderDocs();

  try {
    let text = '';
    if      (type === 'pdf') text = await parsePDF(file);
    else if (type === 'img') text = await parseImage(file);
    else if (type === 'doc') text = await parseDOCX(file);
    else                     text = await parseTXT(file);

    if (!text.trim()) throw new Error('No text could be extracted from this file.');

    const chunks = chunkText(text);
    doc.chunks = chunks.length;
    renderDocs();

    for (let i = 0; i < chunks.length; i++) {
      const embedding = await embed(chunks[i]);
      vectorDB.push({ docId: id, docName: file.name, chunk: chunks[i], embedding, chunkIdx: i });
      doc.progress = Math.round(((i + 1) / chunks.length) * 100);
      renderDocs();
    }
    doc.status = 'ready';
  } catch (e) {
    doc.status = 'error';
    doc.error  = e.message;
  }

  renderDocs();
  updateStats();
}

/* ================================================================
   DOCUMENT LIST RENDER
================================================================ */
const BADGE = { pdf: 'badge-pdf', img: 'badge-img', doc: 'badge-doc', txt: 'badge-txt' };

function renderDocs() {
  document.getElementById('docCount').textContent = docs.length;
  document.getElementById('docList').innerHTML = docs.map(d => `
    <div class="doc-item">
      <div class="doc-name">${d.name}</div>
      <div class="doc-meta">
        <span class="doc-badge ${BADGE[d.type] || 'badge-txt'}">${d.type.toUpperCase()}</span>
        <span class="status-badge ${d.status==='ready'?'sb-ready':d.status==='error'?'sb-error':'sb-processing'}">
          ${d.status==='ready'?'✓ Ready':d.status==='error'?'✗ Error':'⟳ Processing'}
        </span>
        ${d.status==='ready' ? `<span class="doc-chunks">${d.chunks} chunks</span>` : ''}
      </div>
      ${d.status==='processing' ? `<div class="progress-bar"><div class="progress-fill" style="width:${d.progress}%"></div></div>` : ''}
      ${d.status==='error'      ? `<div style="font-size:10px;color:#f87171;margin-top:4px">${d.error}</div>` : ''}
      <button class="doc-del" onclick="deleteDoc('${d.id}')">✕</button>
    </div>
  `).join('');
}

function deleteDoc(id) {
  docs     = docs.filter(d => String(d.id) !== String(id));
  vectorDB = vectorDB.filter(v => String(v.docId) !== String(id));
  renderDocs();
  updateStats();
}

function updateStats() {
  document.getElementById('statChunks').textContent = vectorDB.length;
  document.getElementById('statDocs').textContent   = docs.filter(d => d.status === 'ready').length;
  document.getElementById('statModel').textContent  = (document.getElementById('modelSelect').value || '—').split(':')[0];
}

document.getElementById('modelSelect').addEventListener('change', updateStats);

/* ================================================================
   FILE INPUT & DRAG-DROP
================================================================ */
document.getElementById('fileInput').addEventListener('change', async e => {
  for (const f of Array.from(e.target.files)) await processFile(f);
  e.target.value = '';
});

const uploadArea = document.getElementById('uploadArea');
uploadArea.addEventListener('dragover',  e => { e.preventDefault(); uploadArea.style.borderColor = 'var(--accent)'; });
uploadArea.addEventListener('dragleave', ()  => { uploadArea.style.borderColor = 'var(--border)'; });
uploadArea.addEventListener('drop', async e => {
  e.preventDefault();
  uploadArea.style.borderColor = 'var(--border)';
  for (const f of Array.from(e.dataTransfer.files)) await processFile(f);
});

/* ================================================================
   CHAT
================================================================ */
function clearChat() {
  chatHistory = [];
  document.getElementById('chatArea').innerHTML = `
    <div class="empty-state" id="emptyState">
      <div class="empty-icon">💬</div>
      <div class="empty-title">Ask anything about your documents</div>
      <div class="empty-sub">Upload PDFs, images, text, or Word docs on the left. Then ask questions and the AI will answer using your content.</div>
    </div>`;
}

function appendMsg(role, content, sources) {
  document.getElementById('emptyState')?.remove();
  const ca  = document.getElementById('chatArea');
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  const uniqueSrc = sources ? [...new Set(sources.map(s => s.docName))] : [];
  const srcHtml   = uniqueSrc.length
    ? `<div class="sources">${uniqueSrc.map(n => `<span class="source-chip">📄 ${n}</span>`).join('')}</div>`
    : '';
  div.innerHTML = `
    <div class="avatar">${role === 'user' ? '👤' : '🤖'}</div>
    <div><div class="bubble">${content}</div>${srcHtml}</div>`;
  ca.appendChild(div);
  ca.scrollTop = ca.scrollHeight;
  return div;
}

function showThinking() {
  document.getElementById('emptyState')?.remove();
  const ca  = document.getElementById('chatArea');
  const div = document.createElement('div');
  div.className = 'msg assistant'; div.id = 'thinking';
  div.innerHTML = `<div class="avatar">🤖</div><div class="bubble"><div class="thinking"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div></div>`;
  ca.appendChild(div);
  ca.scrollTop = ca.scrollHeight;
}

async function sendMessage() {
  const inp = document.getElementById('chatInput');
  const q   = inp.value.trim();
  if (!q) return;
  inp.value = ''; inp.style.height = 'auto';

  if (!vectorDB.length) {
    appendMsg('user', q);
    appendMsg('assistant', '⚠️ No documents indexed yet. Please upload a file and wait for processing to complete.');
    return;
  }

  document.getElementById('sendBtn').disabled = true;
  appendMsg('user', q);
  showThinking();

  try {
    const k         = parseInt(document.getElementById('topK').value) || 4;
    const topChunks = await vectorSearch(q, k);
    const context   = topChunks.map((c, i) => `[Source ${i+1}: ${c.docName}]\n${c.chunk}`).join('\n\n---\n\n');
    const model     = document.getElementById('modelSelect').value;

    const systemPrompt = `You are a precise, professional AI assistant. Answer the user's question strictly based on the provided document context below. If the context does not contain sufficient information, say so clearly — do not fabricate answers. Always cite which document your answer comes from.\n\nCONTEXT:\n${context}`;

    const messages = [
      { role: 'system',    content: systemPrompt },
      ...chatHistory,
      { role: 'user',      content: q }
    ];

    const res = await fetch(`${OLLAMA_BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, messages, stream: true })
    });
    if (!res.ok) throw new Error(`Ollama responded with status ${res.status}`);

    document.getElementById('thinking')?.remove();
    const msgDiv = appendMsg('assistant', '', topChunks);
    const bubble = msgDiv.querySelector('.bubble');
    const reader = res.body.getReader();
    const dec    = new TextDecoder();
    let full     = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      for (const line of dec.decode(value).split('\n').filter(Boolean)) {
        try {
          const j = JSON.parse(line);
          if (j.message?.content) {
            full += j.message.content;
            bubble.innerHTML = full.replace(/\n/g, '<br>');
            document.getElementById('chatArea').scrollTop = 99999;
          }
        } catch { /* partial line — skip */ }
      }
    }

    chatHistory.push({ role: 'user',      content: q    });
    chatHistory.push({ role: 'assistant', content: full });
    if (chatHistory.length > 20) chatHistory = chatHistory.slice(-20);

  } catch (e) {
    document.getElementById('thinking')?.remove();
    appendMsg('assistant', `❌ Error: ${e.message}`);
  }

  document.getElementById('sendBtn').disabled = false;
}

/* ── INPUT EVENTS ── */
document.getElementById('chatInput').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
document.getElementById('chatInput').addEventListener('input', function () {
  this.style.height = 'auto';
  this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});

updateStats();