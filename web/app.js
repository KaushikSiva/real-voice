const els = {
  signal: document.querySelector("#signal"),
  messages: document.querySelector("#messages"),
  chatForm: document.querySelector("#chatForm"),
  textInput: document.querySelector("#textInput"),
  sendBtn: document.querySelector("#sendBtn"),
  recordBtn: document.querySelector("#recordBtn"),
  warmBtn: document.querySelector("#warmBtn"),
  referenceForm: document.querySelector("#referenceForm"),
  referenceAudio: document.querySelector("#referenceAudio"),
  referenceText: document.querySelector("#referenceText"),
  modelStatus: document.querySelector("#modelStatus"),
  voiceStatus: document.querySelector("#voiceStatus"),
  timingText: document.querySelector("#timingText"),
  audioQueue: document.querySelector("#audioQueue"),
};

function createSessionId() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return window.crypto.randomUUID();
  }

  if (window.crypto && typeof window.crypto.getRandomValues === "function") {
    const bytes = new Uint8Array(16);
    window.crypto.getRandomValues(bytes);
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    const hex = [...bytes].map((byte) => byte.toString(16).padStart(2, "0")).join("");
    return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
  }

  return `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

const state = {
  sessionId: localStorage.getItem("csm-session-id") || createSessionId(),
  busy: false,
  recording: false,
  playing: false,
  mediaRecorder: null,
  audioParts: [],
  stream: null,
  playbackChain: Promise.resolve(),
};

localStorage.setItem("csm-session-id", state.sessionId);

function setBusy(value) {
  state.busy = value;
  els.sendBtn.disabled = value;
  els.warmBtn.disabled = value;
  els.textInput.disabled = value;
}

function setStatus(text) {
  els.timingText.textContent = text;
}

function addMessage(role, text) {
  const article = document.createElement("article");
  article.className = `message ${role === "You" ? "user" : "assistant"}`;

  const label = document.createElement("span");
  label.textContent = role;

  const body = document.createElement("p");
  body.textContent = text;

  article.append(label, body);
  els.messages.append(article);
  els.messages.scrollTop = els.messages.scrollHeight;
  return article;
}

function appendToMessage(article, text) {
  const body = article.querySelector("p");
  body.textContent += text;
  els.messages.scrollTop = els.messages.scrollHeight;
}

function clearAudioRows() {
  els.audioQueue.replaceChildren();
}

function addAudioRow(url, index, seconds) {
  const row = document.createElement("div");
  row.className = "queue-row";
  const label = document.createElement("span");
  label.textContent = `Chunk ${index}`;
  const link = document.createElement("a");
  link.href = url;
  link.textContent = seconds ? `${seconds}s` : "WAV";
  link.target = "_blank";
  row.append(label, link);
  els.audioQueue.append(row);
}

async function refreshStatus() {
  try {
    const response = await fetch("/api/status");
    const data = await response.json();
    els.modelStatus.textContent = data.tts_loaded ? "Model warm" : "Model cold";
    els.modelStatus.classList.toggle("ready", Boolean(data.tts_loaded));
    els.voiceStatus.textContent = data.voice_locked ? "Voice locked" : "Voice default";
    els.voiceStatus.classList.toggle("ready", Boolean(data.voice_locked));
    if (data.reference_text && !els.referenceText.value) {
      els.referenceText.value = data.reference_text;
    }
  } catch {
    els.modelStatus.textContent = "Server offline";
    els.modelStatus.classList.remove("ready");
  }
}

async function playOne(url) {
  state.playing = true;
  const audio = new Audio(`${url}?t=${Date.now()}`);
  try {
    await audio.play();
    await new Promise((resolve) => {
      audio.onended = resolve;
      audio.onerror = resolve;
    });
  } catch {
    setStatus("Playback blocked");
  }
  state.playing = false;
}

function enqueuePlayback(url) {
  state.playbackChain = state.playbackChain.then(() => playOne(url));
  return state.playbackChain;
}

async function parseEventStream(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() || "";

    for (const rawEvent of events) {
      const lines = rawEvent.split("\n");
      const typeLine = lines.find((line) => line.startsWith("event:"));
      const dataLine = lines.find((line) => line.startsWith("data:"));
      if (!dataLine) continue;
      const type = typeLine ? typeLine.slice(6).trim() : "message";
      const data = JSON.parse(dataLine.slice(5).trim());
      await onEvent(type, data);
    }
  }
}

async function streamText(text, addUserMessage = true) {
  setBusy(true);
  setStatus("Thinking");
  clearAudioRows();
  if (addUserMessage) addMessage("You", text);
  let assistantMessage = null;

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, session_id: state.sessionId }),
    });
    if (!response.ok) throw new Error((await response.json()).detail || response.statusText);

    await parseEventStream(response, async (type, data) => {
      if (type === "start") {
        state.sessionId = data.session_id;
        localStorage.setItem("csm-session-id", state.sessionId);
      } else if (type === "text") {
        if (!assistantMessage) assistantMessage = addMessage("Assistant", "");
        appendToMessage(assistantMessage, data.delta);
      } else if (type === "tts_queued") {
        setStatus(`Voice chunk ${data.index}`);
      } else if (type === "audio") {
        addAudioRow(data.url, data.index, data.seconds);
        setStatus(`Playing chunk ${data.index}`);
        enqueuePlayback(data.url);
      } else if (type === "done") {
        setStatus(`LLM ${data.timings.llm_seconds}s / Voice ${data.timings.tts_seconds}s`);
      } else if (type === "error") {
        throw new Error(data.message || "Request failed.");
      }
    });

    await refreshStatus();
  } catch (error) {
    addMessage("Assistant", error.message || "Request failed.");
    setStatus("Error");
  } finally {
    setBusy(false);
  }
}

els.chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = els.textInput.value.trim();
  if (!text || state.busy) return;
  els.textInput.value = "";
  await streamText(text);
});

els.warmBtn.addEventListener("click", async () => {
  setBusy(true);
  setStatus("Loading CSM");
  try {
    const response = await fetch("/api/warmup", { method: "POST" });
    if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
    const data = await response.json();
    setStatus(`Ready on ${data.device}`);
    await refreshStatus();
  } catch (error) {
    addMessage("Assistant", error.message || "Warmup failed.");
    setStatus("Error");
  } finally {
    setBusy(false);
  }
});

els.referenceForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = els.referenceAudio.files[0];
  const transcript = els.referenceText.value.trim();
  if (state.busy) return;
  if (!file) {
    addMessage("Assistant", "Choose a reference audio file first.");
    setStatus("Voice not saved");
    return;
  }
  if (!transcript) {
    addMessage("Assistant", "Enter the exact words spoken in the reference clip.");
    setStatus("Voice not saved");
    return;
  }

  setBusy(true);
  setStatus("Saving voice");
  const formData = new FormData();
  formData.append("audio", file);
  formData.append("transcript", transcript);

  try {
    const response = await fetch("/api/reference", { method: "POST", body: formData });
    if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
    await refreshStatus();
    setStatus("Voice saved");
    addMessage("Assistant", "Reference voice saved.");
  } catch (error) {
    addMessage("Assistant", error.message || "Voice save failed.");
    setStatus("Error");
  } finally {
    setBusy(false);
  }
});

els.recordBtn.addEventListener("click", async () => {
  if (state.recording && state.mediaRecorder) {
    state.mediaRecorder.stop();
    return;
  }
  if (state.busy) return;

  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    state.audioParts = [];
    state.mediaRecorder = new MediaRecorder(state.stream);
    state.mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) state.audioParts.push(event.data);
    };
    state.mediaRecorder.onstop = sendRecording;
    state.mediaRecorder.start();
    state.recording = true;
    els.recordBtn.classList.add("recording");
    els.recordBtn.textContent = "Stop";
    setStatus("Recording");
  } catch (error) {
    addMessage("Assistant", error.message || "Microphone unavailable.");
  }
});

async function sendRecording() {
  state.recording = false;
  els.recordBtn.classList.remove("recording");
  els.recordBtn.textContent = "Record";
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
  }

  const blob = new Blob(state.audioParts, { type: state.mediaRecorder.mimeType || "audio/webm" });
  const formData = new FormData();
  formData.append("audio", blob, "speech.webm");
  formData.append("session_id", state.sessionId);

  setBusy(true);
  setStatus("Transcribing");
  clearAudioRows();
  let assistantMessage = null;
  try {
    const response = await fetch("/api/talk/stream", { method: "POST", body: formData });
    if (!response.ok) throw new Error((await response.json()).detail || response.statusText);

    await parseEventStream(response, async (type, data) => {
      if (type === "status") {
        setStatus(data.message);
      } else if (type === "transcript") {
        addMessage("You", data.text || "");
        setStatus(`STT ${data.stt_seconds}s`);
      } else if (type === "start") {
        state.sessionId = data.session_id;
        localStorage.setItem("csm-session-id", state.sessionId);
      } else if (type === "text") {
        if (!assistantMessage) assistantMessage = addMessage("Assistant", "");
        appendToMessage(assistantMessage, data.delta);
      } else if (type === "tts_queued") {
        setStatus(`Voice chunk ${data.index}`);
      } else if (type === "audio") {
        addAudioRow(data.url, data.index, data.seconds);
        setStatus(`Playing chunk ${data.index}`);
        enqueuePlayback(data.url);
      } else if (type === "done") {
        setStatus(`LLM ${data.timings.llm_seconds}s / Voice ${data.timings.tts_seconds}s`);
      } else if (type === "error") {
        throw new Error(data.message || "Voice request failed.");
      }
    });

    await refreshStatus();
  } catch (error) {
    addMessage("Assistant", error.message || "Voice request failed.");
    setStatus("Error");
  } finally {
    setBusy(false);
  }
}

function drawSignal() {
  const canvas = els.signal;
  const ctx = canvas.getContext("2d");
  const ratio = window.devicePixelRatio || 1;
  const width = Math.floor(window.innerWidth * ratio);
  const height = Math.floor(window.innerHeight * ratio);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  const time = performance.now() / 1000;
  ctx.clearRect(0, 0, width, height);
  ctx.lineWidth = ratio;

  const active = state.recording || state.playing || state.busy;
  const amplitude = active ? 56 * ratio : 22 * ratio;
  const baseY = height * 0.52;
  const spacing = Math.max(18 * ratio, width / 80);

  for (let x = 0; x < width; x += spacing) {
    const wave = Math.sin(x * 0.006 + time * (active ? 3.2 : 0.9));
    const heightScale = amplitude * (0.35 + Math.abs(wave));
    const alpha = active ? 0.22 : 0.11;
    ctx.strokeStyle = `rgba(87, 227, 137, ${alpha})`;
    ctx.beginPath();
    ctx.moveTo(x, baseY - heightScale);
    ctx.lineTo(x, baseY + heightScale);
    ctx.stroke();
  }

  requestAnimationFrame(drawSignal);
}

refreshStatus();
drawSignal();
