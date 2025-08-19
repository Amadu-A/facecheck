// verification/static/verification/base.js
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const startBtn = document.getElementById('startCam');
const shotBtn = document.getElementById('shot');
const captureForm = document.getElementById('captureForm');
const imageDataInput = document.getElementById('image_data');

let stream = null;

if (startBtn) {
  startBtn.addEventListener('click', async () => {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      video.srcObject = stream;
      shotBtn.disabled = false;
    } catch (e) {
      alert('Не удалось открыть камеру: ' + e);
    }
  });
}

if (shotBtn) {
  shotBtn.addEventListener('click', () => {
    if (!stream) return;
    const w = video.videoWidth;
    const h = video.videoHeight;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, w, h);
    const dataUrl = canvas.toDataURL('image/png');
    imageDataInput.value = dataUrl;
    captureForm.submit();
  });
}
