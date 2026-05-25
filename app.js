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
   INDEXEDDB PERSISTENCE LAYER
================================================================ */
const DB_NAME = 'LocalRAG_DB';
const DB_VERSION = 1;

/**
 * Establishes a connection to IndexedDB, creating tables if initialized for the first time.
 */
function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = (e) => {
      const db = e.target.result;
      if (!db.objectStoreNames.contains('docs')) {
        db.createObjectStore('docs', { keyPath: 'id' });
      }
      if (!db.objectStoreNames.contains('vectorDB')) {
        db.createObjectStore('vectorDB', { autoIncrement: true });
      }
    };
    req.onsuccess = (e) => resolve(e.target.result);
    req.onerror = (e) => reject(e.target.error);
  });
}

/**
 * Saves or updates a document metadata record in the local database.
 */
async function saveDoc(doc) {
  try {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction('docs', 'readwrite');
      tx.objectStore('docs').put(doc);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  } catch (err) {
    console.error('Failed to save document metadata in database:', err);
  }
}

/**
 * Pulls all saved document metadata records from the database.
 */
async function loadDocs() {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('docs', 'readonly');
    const req = tx.objectStore('docs').getAll();
    req.onsuccess = () => resolve(req.result || []);
    req.onerror = () => reject(tx.error);
  });
}

/**
 * Adds an array of generated text chunk vectors to the database store.
 */
async function saveVectors(vectors) {
  try {
    const db = await openDB();
    const tx = db.transaction('vectorDB', 'readwrite');
    const store = tx.objectStore('vectorDB');
    for (const v of vectors) {
      store.add(v);
    }
    return new Promise((resolve) => tx.oncomplete = () => resolve());
  } catch (err) {
    console.error('Failed to save generated vectors in database:', err);
  }
}

/**
 * Pulls all indexed vector items from the database store.
 */
async function loadVectors() {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('vectorDB', 'readonly');
    const req = tx.objectStore('vectorDB').getAll();
    req.onsuccess = () => resolve(req.result || []);
    req.onerror = () => reject(tx.error);
  });
}

/**
 * Removes a document record and its associated embeddings from the database.
 */
async function deleteDocFromDB(id) {
  try {
    const db = await openDB();
    const tx = db.transaction(['docs', 'vectorDB'], 'readwrite');
    tx.objectStore('docs').delete(id);
    
    const vectorStore = tx.objectStore('vectorDB');
    const req = vectorStore.openCursor();
    req.onsuccess = (e) => {
      const cursor = e.target.result;
      if (cursor) {
        if (String(cursor.value.docId) === String(id)) {
          cursor.delete();
        }
        cursor.continue();
      }
    };
    return new Promise((resolve) => tx.oncomplete = () => resolve());
  } catch (err) {
    console.error('Failed to purge document records from database:', err);
  }
}

/* ================================================================
   MODEL TRANSLATION MAP & NORMALIZATION
================================================================ */
const GEMINI_MODEL_MAP = {
  '3.5': 'gemini-1.5-pro',
  '3.5 pro': 'gemini-1.5-pro',
  '3.5-pro': 'gemini-1.5-pro',
  '3.5 flash': 'gemini-1.5-flash',
  '3.5-flash': 'gemini-1.5-flash',
  'pro': 'gemini-1.5-pro',
  'flash': 'gemini-1.5-flash',
  
  'gemini-1.5-flash': 'gemini-1.5-flash',
  'gemini-1.5-pro': 'gemini-1.5-pro',
  'gemini-2.5-flash': 'gemini-2.5-flash',
  'gemini-2.5-pro': 'gemini-2.5-pro'
};

/**
 * Matches custom user input strings to supported Google API models.
 */
function getNormalizedGeminiModel(userVal) {
  const normalized = String(userVal).trim().toLowerCase();
  if (GEMINI_MODEL_MAP[normalized]) {
    return GEMINI_MODEL_MAP[normalized];
  }
  if (normalized.startsWith('gemini-')) {
    return normalized;
  }
  return 'gemini-1.5-flash';
}

/* ================================================================
   DYNAMIC PROVIDER INTERFACE CONTROL
================================================================ */
function toggleProviderUI() {
  const provider = document.getElementById('providerSelect').value;
  const apiKeyWrap = document.getElementById('apiKeyWrap');
  const modelSelect = document.getElementById('modelSelect');
  const statusArea = document.getElementById('statusArea');

  if (provider === 'gemini') {
    apiKeyWrap.style.display = 'block';
    statusArea.style.display = 'none';
    
    modelSelect.innerHTML = `
      <option value="gemini-2.5-flash">Gemini 2.5 Flash (Recommended)</option>
      <option value="gemini-1.5-flash">Gemini 1.5 Flash (Legacy Speed)</option>
      <option value="gemini-1.5-pro">Gemini 1.5 Pro (Legacy Analytical)</option>
      <option value="3.5">Gemini 3.5 (Custom Selection)</option>
      <option value="3.5-pro">Gemini 3.5 Pro (Custom Selection)</option>
    `;
  } else if (provider === 'dryrun') {
    apiKeyWrap.style.display = 'none';
    statusArea.style.display = 'none';
    modelSelect.innerHTML = `<option value="dry-run">No LLM (Diagnostics Only)</option>`;
  } else {
    // Ollama Local Mode
    apiKeyWrap.style.display = 'none';
    statusArea.style.display = 'flex';
    modelSelect.innerHTML = `
      <option value="llama3.2">llama3.2</option>
      <option value="gemma2">gemma2</option>
    `;
    checkOllama();
  }
  updateStats();
}

/* ================================================================
   OLLAMA CONNECTION CHECK
================================================================ */
async function checkOllama() {
  const provider = document.getElementById('providerSelect').value;
  if (provider !== 'ollama') return;

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
  const provider = document.getElementById('providerSelect').value;
  if (provider !== 'ollama') return;

  const sel = document.getElementById('modelSelect');
  const currentVal = sel.value;
  sel.innerHTML = '';
  models.forEach(m => {
    const o = document.createElement('option');
    o.value = m.name; o.textContent = m.name;
    sel.appendChild(o);
  });
  if (currentVal && Array.from(sel.options).some(o => o.value === currentVal)) {
    sel.value = currentVal;
  }
  updateStats();
}

/* ================================================================
   EMBEDDING GENERATION (OLLAMA VS GEMINI VS DRYRUN)
================================================================ */
async function embed(text) {
  const provider = document.getElementById('providerSelect').value;

  if (provider === 'gemini') {
    const key = document.getElementById('apiKeyInput').value.trim();
    if (!key) throw new Error('Google Gemini API Key is missing.');

    try {
      // FIX: Use Google's newly mandated 2026 embedding model
      return await fetchGeminiEmbedding(key, 'gemini-embedding-001', text);
    } catch (error) {
       throw new Error(`Gemini Embedding Failed: ${error.message}`);
    }

  } else if (provider === 'dryrun') {
    return Array.from({ length: 768 }, () => Math.random() - 0.5);
  } else {
    // Local Ollama
    const r = await fetch(`${OLLAMA_BASE}/api/embeddings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: EMBED_MODEL, prompt: text })
    });
    if (!r.ok) throw new Error('Ollama embedding failed.');
    return (await r.json()).embedding;
  }
}

/**
 * Standardized Google Gemini API Fetch
 */
async function fetchGeminiEmbedding(apiKey, modelName, text) {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${modelName}:embedContent?key=${apiKey}`;

  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      content: { parts: [{ text: text }] },
      // taskType helps Google optimize the vector for document retrieval
      taskType: "RETRIEVAL_DOCUMENT" 
    })
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.error?.message || `HTTP ${response.status}`);
  }

  const data = await response.json();
  if (!data.embedding || !data.embedding.values) {
    throw new Error('Empty response returned by Gemini API.');
  }
  return data.embedding.values;
}
/* ================================================================
   VECTOR SEARCH ENGINE (DENSE RETRIEVAL)
================================================================ */
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
   KEYWORD SEARCH ENGINE & HYBRID RRF MERGER
================================================================ */
function keywordSearch(query, k_limit) {
  const terms = query.toLowerCase().match(/\w+/g) || [];
  if (terms.length === 0) return [];
  
  const scores = vectorDB.map(c => {
    const text = c.chunk.toLowerCase();
    let occurrences = 0;
    terms.forEach(term => {
      let idx = text.indexOf(term);
      while (idx !== -1) {
        occurrences++;
        idx = text.indexOf(term, idx + 1);
      }
    });
    const totalWords = text.split(/\s+/).length || 1;
    return { chunk: c, score: occurrences / totalWords };
  });

  return scores
    .filter(s => s.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, k_limit)
    .map(s => s.chunk);
}

async function hybridSearch(query, k) {
  if (!vectorDB.length) return [];
  
  const denseResults = await vectorSearch(query, vectorDB.length);
  const denseRanked = [...denseResults].sort((a, b) => b.score - a.score);
  const sparseRanked = keywordSearch(query, vectorDB.length);

  const rrfConstant = 60;
  const scoreMap = new Map();

  const addRRFScore = (chunk, rank) => {
    const key = `${chunk.docId}-${chunk.chunkIdx}`;
    if (!scoreMap.has(key)) {
      scoreMap.set(key, { chunk, score: 0 });
    }
    scoreMap.get(key).score += (1.0 / (rrfConstant + rank));
  };

  denseRanked.forEach((item, index) => { addRRFScore(item, index + 1); });
  sparseRanked.forEach((item, index) => { addRRFScore(item, index + 1); });

  return Array.from(scoreMap.values())
    .sort((a, b) => b.score - a.score)
    .slice(0, k)
    .map(entry => entry.chunk);
}

/* ================================================================
   PARENT-CHILD CONTEXT EXPANSION
================================================================ */
function getExpandedContext(retrievedChunk, overlapBefore = 1, overlapAfter = 1) {
  const siblings = vectorDB.filter(c => String(c.docId) === String(retrievedChunk.docId));
  siblings.sort((a, b) => a.chunkIdx - b.chunkIdx);

  const targetIdx = retrievedChunk.chunkIdx;
  const startIdx = Math.max(0, targetIdx - overlapBefore);
  const endIdx = Math.min(siblings.length - 1, targetIdx + overlapAfter);

  const expandedChunks = siblings.slice(startIdx, endIdx + 1);
  const combinedText = expandedChunks.map(c => c.chunk).join(' ');

  return {
    text: combinedText,
    chunkIndices: expandedChunks.map(c => c.chunkIdx)
  };
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
  await saveDoc(doc);

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
    await saveDoc(doc);

    const chunkVectors = [];
    for (let i = 0; i < chunks.length; i++) {
      const embedding = await embed(chunks[i]);
      const vItem = { docId: id, docName: file.name, chunk: chunks[i], embedding, chunkIdx: i };
      vectorDB.push(vItem);
      chunkVectors.push(vItem);
      doc.progress = Math.round(((i + 1) / chunks.length) * 100);
      renderDocs();
    }
    
    await saveVectors(chunkVectors);
    doc.status = 'ready';
    await saveDoc(doc);
  } catch (e) {
    doc.status = 'error';
    doc.error  = e.message;
    await saveDoc(doc);
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

async function deleteDoc(id) {
  const matchId = String(id);
  docs     = docs.filter(d => String(d.id) !== matchId);
  vectorDB = vectorDB.filter(v => String(v.docId) !== matchId);
  
  const numericId = isNaN(id) ? id : Number(id);
  await deleteDocFromDB(numericId);
  
  renderDocs();
  updateStats();
}

function updateStats() {
  document.getElementById('statChunks').textContent = vectorDB.length;
  document.getElementById('statDocs').textContent   = docs.filter(d => d.status === 'ready').length;
  
  const prov = document.getElementById('providerSelect').value;
  if (prov === 'dryrun') {
    document.getElementById('statModel').textContent = 'Dry Run';
  } else {
    document.getElementById('statModel').textContent = (document.getElementById('modelSelect').value || '—').split(':')[0];
  }
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
   CHAT VIEW MANIPULATION
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

/* ================================================================
   CHAT REQUEST & EXECUTION (MULTI-PROVIDER ROUTER)
================================================================ */
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
    const k = parseInt(document.getElementById('topK').value) || 4;
    
    // Select retrieval strategy
    const isHybridEnabled = document.getElementById('hybridSearchToggle').checked;
    const retrievedChunks = isHybridEnabled ? await hybridSearch(q, k) : await vectorSearch(q, k);

    // Apply parent-child context expansion
    const isExpansionEnabled = document.getElementById('contextExpansionToggle').checked;
    const contextBlocks = [];

    retrievedChunks.forEach(c => {
      if (isExpansionEnabled) {
        const expanded = getExpandedContext(c, 1, 1);
        contextBlocks.push(`[Source: ${c.docName} (Chunks ${expanded.chunkIndices.join(', ')})]\n${expanded.text}`);
      } else {
        contextBlocks.push(`[Source: ${c.docName} (Chunk ${c.chunkIdx})]\n${c.chunk}`);
      }
    });

    const context = contextBlocks.join('\n\n---\n\n');
    const provider = document.getElementById('providerSelect').value;
    const rawModelSelection = document.getElementById('modelSelect').value;

    const systemPrompt = `You are a precise, professional AI assistant. Answer the user's question strictly based on the provided document context below. If the context does not contain sufficient information, say so clearly — do not fabricate answers. Always cite which document your answer comes from.\n\nCONTEXT:\n${context}`;

    /* ── DIRECT DIAGNOSTICS MODE (DRY RUN) ── */
    if (provider === 'dryrun') {
      document.getElementById('thinking')?.remove();
      
      const debugResultHTML = `
        <strong>🔍 DIAGNOSTIC MODE: RAG Pipeline Execution Test</strong><br><br>
        <strong>Your Query:</strong> "${q}"<br><br>
        <strong>Retrieved ${retrievedChunks.length} Context Chunks:</strong><br>
        ${retrievedChunks.map((c, i) => `
          <div style="margin-top:8px; padding:6px; background:rgba(255,255,255,0.05); border-radius:4px; font-size:11px;">
            <b>Chunk #${i+1} [${c.docName} - Index ${c.chunkIdx}]:</b><br>
            <i>"${c.chunk.substring(0, 180)}..."</i>
          </div>
        `).join('')}
        <br>
        <strong>Compiled System Prompt payload:</strong>
        <pre style="white-space:pre-wrap; font-size:10px; background:#111; padding:8px; border-radius:4px; border:1px solid #333; margin-top:8px; max-height:150px; overflow-y:auto;">${systemPrompt}</pre>
      `;
      appendMsg('assistant', debugResultHTML, retrievedChunks);
      document.getElementById('sendBtn').disabled = false;
      return;
    }

    /* ── GOOGLE GEMINI API FLOW ── */
    if (provider === 'gemini') {
      const key = document.getElementById('apiKeyInput').value.trim();
      if (!key) throw new Error('Gemini API Key is missing. Add your API Key to the top bar input field.');

      const mappedModel = getNormalizedGeminiModel(rawModelSelection);
      const contents = [];
      
      // Setup context guidelines
      contents.push({
        role: 'user',
        parts: [{ text: systemPrompt }]
      });
      contents.push({
        role: 'model',
        parts: [{ text: "Understood. I will restrict my responses strictly to the facts provided within the given document context." }]
      });

      // Insert message history
      chatHistory.forEach(h => {
        contents.push({
          role: h.role === 'user' ? 'user' : 'model',
          parts: [{ text: h.content }]
        });
      });

      // Add the current query
      contents.push({
        role: 'user',
        parts: [{ text: q }]
      });

      const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${mappedModel}:generateContent?key=${key}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ contents: contents })
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.error?.message || `Gemini API responded with status ${res.status}`);
      }

      const data = await res.json();
      const outputText = data.candidates?.[0]?.content?.parts?.[0]?.text || 'No response text generated.';

      document.getElementById('thinking')?.remove();
      appendMsg('assistant', outputText.replace(/\n/g, '<br>'), retrievedChunks);

      chatHistory.push({ role: 'user',      content: q    });
      chatHistory.push({ role: 'assistant', content: outputText });
      if (chatHistory.length > 20) chatHistory = chatHistory.slice(-20);

    } else {
      /* ── LOCAL OLLAMA FLOW ── */
      const messages = [
        { role: 'system',    content: systemPrompt },
        ...chatHistory,
        { role: 'user',      content: q }
      ];

      const res = await fetch(`${OLLAMA_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: rawModelSelection, messages, stream: true })
      });
      if (!res.ok) throw new Error(`Ollama responded with status ${res.status}`);

      document.getElementById('thinking')?.remove();
      const msgDiv = appendMsg('assistant', '', retrievedChunks);
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
          } catch { /* parse chunk skip */ }
        }
      }

      chatHistory.push({ role: 'user',      content: q    });
      chatHistory.push({ role: 'assistant', content: full });
      if (chatHistory.length > 20) chatHistory = chatHistory.slice(-20);
    }

  } catch (e) {
    document.getElementById('thinking')?.remove();
    appendMsg('assistant', `❌ Error: ${e.message}`);
  }

  document.getElementById('sendBtn').disabled = false;
}

/* ── INPUT EVENT HANDLERS ── */
document.getElementById('chatInput').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
document.getElementById('chatInput').addEventListener('input', function () {
  this.style.height = 'auto';
  this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});

/* ================================================================
   APPLICATION SYSTEM INITIALIZATION
================================================================ */
async function initApp() {
  try {
    docs = await loadDocs();
    vectorDB = await loadVectors();
    renderDocs();
    updateStats();
  } catch (err) {
    console.warn('Unable to restore database state from storage:', err);
  }
  
  // Set default UI state
  toggleProviderUI();
}

initApp();