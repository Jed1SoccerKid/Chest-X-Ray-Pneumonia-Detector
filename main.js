const CLASS_LABELS = ['NORMAL', 'PNEUMONIA'];
const IMG_SIZE     = 224;
const MODEL_PATH   = 'ONNXmodels/chest_xray_model.onnx';

const dropZone       = document.getElementById('drop-zone');
const fileInput      = document.getElementById('file-input');
const resultSection  = document.getElementById('result-section');
const previewImg     = document.getElementById('preview-img');
const loadingState   = document.getElementById('loading-state');
const diagnosisEl    = document.getElementById('diagnosis');
const diagnosisIcon  = document.getElementById('diagnosis-icon');
const diagnosisLabel = document.getElementById('diagnosis-label');
const confValue      = document.getElementById('conf-value');
const barFill        = document.getElementById('bar-fill');
const probBreakdown  = document.getElementById('prob-breakdown');
const normalBar      = document.getElementById('normal-bar');
const pneumoniaBar   = document.getElementById('pneumonia-bar');
const normalPct      = document.getElementById('normal-pct');
const pneumoniaPct   = document.getElementById('pneumonia-pct');
const resetBtn       = document.getElementById('reset-btn');

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) handleFile(file);
});

dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', e => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});

resetBtn.addEventListener('click', () => {
  resultSection.style.display = 'none';
  dropZone.parentElement.style.display = 'block';
  fileInput.value = '';
});

function handleFile(file) {
  const reader = new FileReader();
  reader.onload = async (e) => {
    const dataURL = e.target.result;

    dropZone.parentElement.style.display = 'none';
    resultSection.style.display = 'grid';
    loadingState.style.display  = 'flex';
    diagnosisEl.style.display   = 'none';
    probBreakdown.style.display = 'none';

    previewImg.src = dataURL;

    try {
      const tensor = await imageToTensor(dataURL);
      const { probs, predictedIdx } = await runInference(tensor);
      showResult(probs, predictedIdx);
    } catch (err) {
      console.error('Inference error:', err);
      loadingState.innerHTML = `<p style="color:#ff4d6d">Error running model.<br><small>${err.message}</small></p>`;
    }
  };
  reader.readAsDataURL(file);
}


function imageToTensor(dataURL) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas  = document.createElement('canvas');
      canvas.width  = IMG_SIZE;
      canvas.height = IMG_SIZE;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, IMG_SIZE, IMG_SIZE);

      const imageData = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
      const { data }  = imageData; // RGBA, length = 224*224*4

    
      const floatArr = new Float32Array(1 * 1 * IMG_SIZE * IMG_SIZE);

      for (let i = 0; i < IMG_SIZE * IMG_SIZE; i++) {
        const r = data[i * 4];
        const g = data[i * 4 + 1];
        const b = data[i * 4 + 2];
        const gray = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
        floatArr[i] = (gray - 0.5) / 0.5;  // normalize: [-1, 1]
      }

      const tensor = new ort.Tensor('float32', floatArr, [1, 1, IMG_SIZE, IMG_SIZE]);
      resolve(tensor);
    };
    img.onerror = reject;
    img.src = dataURL;
  });
}

let _session = null;  // cache session so we only load once

async function runInference(tensor) {
  if (!_session) {
    _session = await ort.InferenceSession.create(MODEL_PATH);
  }

  const feeds     = { input1: tensor };
  const outputMap = await _session.run(feeds);
  const logits    = Array.from(outputMap.output1.data);  // [logit_normal, logit_pneumonia]

  const maxLogit = Math.max(...logits);
  const exps     = logits.map(l => Math.exp(l - maxLogit));
  const sumExps  = exps.reduce((a, b) => a + b, 0);
  const probs    = exps.map(e => e / sumExps);

  const predictedIdx = probs.indexOf(Math.max(...probs));
  return { probs, predictedIdx };
}

function showResult(probs, predictedIdx) {
  loadingState.style.display = 'none';

  const label   = CLASS_LABELS[predictedIdx];
  const conf    = probs[predictedIdx];
  const confPct = (conf * 100).toFixed(1) + '%';

  const isNormal = predictedIdx === 0;

  diagnosisIcon.textContent  = isNormal ? '✅' : '⚠️';
  diagnosisLabel.textContent = label;
  diagnosisLabel.className   = 'diagnosis-label ' + (isNormal ? 'is-normal' : 'is-pneumonia');

  confValue.textContent = confPct;
  barFill.style.width   = confPct;
  barFill.className     = 'bar-fill ' + (isNormal ? 'normal' : 'pneumonia');

  diagnosisEl.style.display = 'block';

  const nPct = (probs[0] * 100).toFixed(1);
  const pPct = (probs[1] * 100).toFixed(1);

  normalPct.textContent    = nPct + '%';
  pneumoniaPct.textContent = pPct + '%';

  setTimeout(() => {
    normalBar.style.width    = nPct + '%';
    pneumoniaBar.style.width = pPct + '%';
  }, 100);

  probBreakdown.style.display = 'block';
}
