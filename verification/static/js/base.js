// verification/static/verification/base.js
// ===== утилиты =====
function getCookie(name) {
  const value = `; ${document.cookie}`;
  const parts = value.split(`; ${name}=`);
  if (parts.length === 2) return decodeURIComponent(parts.pop().split(';').shift());
  return null;
}

const CSRF = getCookie('csrftoken');
const dbg = document.getElementById('debug');
function printDbg(msg, data) {
  try {
    const line = `[${new Date().toLocaleTimeString()}] ${msg}` + (data ? ` ${JSON.stringify(data)}` : '');
    if (dbg) {
      dbg.textContent += line + "\n";
      dbg.scrollTop = dbg.scrollHeight;
    }
    // отправляем на сервер
    fetch('/client-log/', {
      method: 'POST',
      headers: {'Content-Type': 'application/json', 'X-CSRFToken': CSRF || ''},
      body: JSON.stringify({ event: msg, details: data || null, ts: Date.now() })
    }).catch(()=>{});
  } catch(e) {/*no-op*/}
}

// ===== элементы =====
const video   = document.getElementById('video');
const canvas  = document.getElementById('canvas');
const startBtn= document.getElementById('startCam');
const shotBtn = document.getElementById('shot');
const checkBtn= document.getElementById('checkCam');
const captureForm    = document.getElementById('captureForm');
const imageDataInput = document.getElementById('image_data');

let stream = null;

// ===== первичная диагностика клиентских возможностей =====
(function initialProbe() {
  const secure = window.isSecureContext || location.hostname === 'localhost' || location.hostname === '127.0.0.1';
  printDbg('page_load', { userAgent: navigator.userAgent, secureContext: !!secure });
  if (!secure) {
    printDbg('warning_insecure_context', { note: 'getUserMedia требует HTTPS или localhost' });
  }
  const hasMedia = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  printDbg('media_devices_support', { hasMedia });

  if (navigator.permissions && navigator.permissions.query) {
    navigator.permissions.query({ name: 'camera' }).then((st) => {
      printDbg('permission_camera_state', { state: st.state });
      st.onchange = () => printDbg('permission_camera_state_change', { state: st.state });
    }).catch((e)=> printDbg('permission_query_error', { error: String(e) }));
  } else {
    printDbg('permission_api_absent');
  }

  if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
    navigator.mediaDevices.enumerateDevices().then(devs => {
      const cams = devs.filter(d => d.kind === 'videoinput').map(d => ({deviceId: d.deviceId, label: d.label}));
      printDbg('enumerate_devices', { videoInputs: cams });
    }).catch(e => printDbg('enumerate_devices_error', { error: String(e) }));
  }
})();

// ===== кнопка "Включить камеру" =====
if (startBtn) {
  startBtn.addEventListener('click', async () => {
    printDbg('start_camera_click');
    if (!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) {
      printDbg('getUserMedia_unsupported');
      alert('Ваш браузер не поддерживает getUserMedia.');
      return;
    }
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: {ideal:1280}, height:{ideal:720} }, audio: false });
      video.srcObject = stream;
      shotBtn.disabled = false;
      printDbg('getUserMedia_success');

      const track = stream.getVideoTracks()[0];
      printDbg('track_info', {
        label: track.label,
        settings: track.getSettings ? track.getSettings() : null,
        capabilities: track.getCapabilities ? track.getCapabilities() : null,
        readyState: track.readyState
      });

      video.onloadedmetadata = () => printDbg('video_loadedmetadata', { width: video.videoWidth, height: video.videoHeight });
      video.onplay = () => printDbg('video_play');
      video.onpause = () => printDbg('video_pause');
      video.onended = () => printDbg('video_ended');

    } catch (e) {
      const err = e && e.name ? `${e.name}: ${e.message}` : String(e);
      printDbg('getUserMedia_error', { error: err });
      alert('Не удалось открыть камеру: ' + err);
    }
  });
}

// ===== кнопка "Сделать фото" =====
if (shotBtn) {
  shotBtn.addEventListener('click', () => {
    if (!stream) {
      printDbg('shot_click_no_stream');
      return;
    }
    const w = video.videoWidth;
    const h = video.videoHeight;
    if (!w || !h) {
      printDbg('shot_click_no_dims');
      return;
    }
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, w, h);
    const dataUrl = canvas.toDataURL('image/png');
    imageDataInput.value = dataUrl;
    printDbg('shot_captured', { w, h });
    captureForm.submit();
  });
}

// ===== кнопка "Проверить камеру" (серверные проверки Linux) =====
if (checkBtn) {
  checkBtn.addEventListener('click', async () => {
    printDbg('diagnostics_click');
    try {
      const resp = await fetch('/diagnostics/', {
        method: 'GET',
        headers: {'Accept': 'application/json'},
      });
      const data = await resp.json();
      printDbg('diagnostics_result', data);
    } catch (e) {
      printDbg('diagnostics_error', { error: String(e) });
    }
  });
}