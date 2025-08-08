// RF client-side inference
let MODEL = null, IMPUTER = null, SCALER = null;
let CLASS_LABELS = null, FEATURES = null;

const el = (id) => document.getElementById(id);
const loader = el('loader');
const formCard = el('formCard');
const resultCard = el('resultCard');
const batchCard = el('batchCard');
const formDiv = el('form');
const loadMsg = el('loadMsg');
const predLabel = el('predLabel');
const probsDiv = el('probs');

// ---------- Utils ----------
function argmax(arr){ let m=-Infinity, idx=-1; for(let i=0;i<arr.length;i++){ if(arr[i]>m){ m=arr[i]; idx=i;} } return idx; }
function softmax(arr){ const m = Math.max(...arr); const exps = arr.map(v=>Math.exp(v-m)); const s=exps.reduce((a,b)=>a+b,0); return exps.map(v=>v/s); }
function mean(arr){ return arr.reduce((a,b)=>a+b,0) / arr.length; }
function isNum(x){ return typeof x==='number' && !Number.isNaN(x) && Number.isFinite(x); }

// ---------- Load artifacts ----------
async function loadArtifacts(){
  try{
    loadMsg.textContent = 'Loading...';
    const [model, imputer, scaler] = await Promise.all([
      fetch('model.json').then(r=>r.json()),
      fetch('imputer.json').then(r=>r.json()),
      fetch('scaler.json').then(r=>r.json())
    ]);
    MODEL = model;
    IMPUTER = imputer;
    SCALER = scaler;
    FEATURES = model.feature_names;
    CLASS_LABELS = Array.from({length: model.n_classes}, (_,i)=>String(i)); // "0","1","2"

    // build form
    formDiv.innerHTML = '';
    FEATURES.forEach(f=>{
      const wrap = document.createElement('div');
      wrap.className = 'row';
      wrap.innerHTML = `
        <div>
          <label>${f}</label>
          <input inputmode="decimal" type="text" id="feat_${f}" placeholder="number" />
        </div>
      `;
      formDiv.appendChild(wrap);
    });
    el('featList').textContent = FEATURES.join(', ');
    loader.classList.add('hidden');
    formCard.classList.remove('hidden');
    batchCard.classList.remove('hidden');
    loadMsg.textContent = 'Loaded.';
  }catch(e){
    loadMsg.textContent = 'Failed to load artifacts. Place JSON files next to index.html.';
    console.error(e);
  }
}

el('loadBtn').addEventListener('click', loadArtifacts);

// ---------- Preprocess ----------
function preprocess(sampleObj){
  // Returns standardized feature vector aligned with FEATURES
  return FEATURES.map(feat => {
    let v = Number(sampleObj[feat]);
    if (!isNum(v)) v = Number(IMPUTER[feat]);
    let m = SCALER.mean[feat];
    let sd = SCALER.sd[feat];
    if (!isNum(m)) m = 0;
    if (!isNum(sd) || sd === 0) sd = 1;
    return (v - m) / sd;
  });
}

// ---------- Tree traversal ----------
function traverseTree(tree, sampleObj){
  if ('value' in tree){ return tree.value.slice(); } // copy one-hot
  const feat = tree.feature;
  const thr = tree.threshold;
  const v = Number(sampleObj[feat]);
  const goLeft = isNum(v) ? (v <= thr) : true; // if NaN, default left
  return traverseTree(goLeft ? tree.left : tree.right, sampleObj);
}

function predictOne(sampleObj){
  // In your R script, RF was trained on standardized features.
  // Therefore, traverse trees using standardized values.
  const z = {};
  const zvec = preprocess(sampleObj);
  FEATURES.forEach((f,i)=>{ z[f] = zvec[i]; });

  const probs = Array(MODEL.n_classes).fill(0);
  for (const t of MODEL.trees){
    const leaf = traverseTree(t, z); // leaf is one-hot vector length n_classes
    for (let i=0;i<probs.length;i++) probs[i] += leaf[i];
  }
  for (let i=0; i<probs.length; i++) probs[i] /= MODEL.trees.length;
  return probs;
}

// ---------- UI actions ----------
el('predictBtn').addEventListener('click', () => {
  const row = {};
  FEATURES.forEach(f => { row[f] = el(`feat_${f}`).value.trim(); });
  const probs = predictOne(row);
  const idx = argmax(probs);
  predLabel.textContent = CLASS_LABELS[idx];
  probsDiv.textContent = 'Probabilities: [' + probs.map(p=>p.toFixed(4)).join(', ') + ']';
  resultCard.classList.remove('hidden');
});

el('clearBtn').addEventListener('click', () => {
  FEATURES.forEach(f => { el(`feat_${f}`).value = ''; });
  predLabel.textContent = '-';
  probsDiv.textContent = '';
});

// ---------- CSV batch prediction ----------
function parseCSV(text){
  const lines = text.split(/\r?\n/).filter(Boolean);
  const header = lines[0].split(',').map(s=>s.trim());
  const rows = lines.slice(1).map(line => {
    const cols = line.split(',').map(s=>s.trim());
    const obj = {};
    header.forEach((h,i)=>{ obj[h] = cols[i]; });
    return obj;
  });
  return { header, rows };
}

function toCSV(rows){
  const header = ['pred', 'prob_0','prob_1','prob_2', ...FEATURES];
  const lines = [header.join(',')];
  for (const r of rows){
    const arr = [r.pred, ...r.probs.map(x=>x.toFixed(6)), ...FEATURES.map(f=>r.raw[f] ?? '')];
    lines.push(arr.join(','));
  }
  return lines.join('\n');
}

document.getElementById('runBatch').addEventListener('click', async () => {
  const f = document.getElementById('csvFile').files[0];
  if (!f){ alert('Choose a CSV file'); return; }
  const text = await f.text();
  const { header, rows } = parseCSV(text);
  // check columns
  const missing = FEATURES.filter(f => !header.includes(f));
  if (missing.length){
    alert('CSV missing columns: ' + missing.join(', '));
    return;
  }
  const out = [];
  for (const r of rows){
    const probs = predictOne(r);
    const predIdx = argmax(probs);
    out.push({ pred: String(predIdx), probs, raw: r });
  }
  const csv = toCSV(out);
  const blob = new Blob([csv], {type: 'text/csv'});
  const url = URL.createObjectURL(blob);
  const link = document.getElementById('dlLink');
  link.href = url;
  link.classList.remove('hidden');
  link.textContent = 'Download predictions.csv';
});
