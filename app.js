/* =========================
   Core Layer
   ========================= */
const MathKit = {
  sigmoid: x => 1/(1+Math.exp(-x)),
  logit:   p => Math.log(p/(1-p)),
  argmax(a){ let m=-Infinity, mi=-1; for(let i=0;i<a.length;i++){ if(a[i]>m){m=a[i];mi=i;} } return mi; },
  rowNormalize(arr){ const s = arr.reduce((a,b)=>a+b,0)||1; return arr.map(v=>v/s); },
  softmaxNegDist(dists, alpha){
    const exps = dists.map(d => Math.exp(-alpha*d));
    const s = exps.reduce((a,b)=>a+b,0)||1;
    return exps.map(v=>v/s);
  }
};

const CSVKit = {
  parse(text){
    const lines = text.replace(/\r/g,'').split('\n').filter(x=>x.trim().length>0);
    if(!lines.length) return {header:[],rows:[]};
    const header = lines[0].split(',').map(h=>h.trim());
    const rows = [];
    for(let i=1;i<lines.length;i++){
      const cols = lines[i].split(','); // simplified: no complex escaping
      const rec = {};
      header.forEach((h,idx)=>{ rec[h] = (cols[idx]!==undefined? cols[idx].trim() : ''); });
      rows.push(rec);
    }
    return {header, rows};
  },
  toCSV(arr){
    if(!arr.length) return '';
    const cols = Object.keys(arr[0]);
    const head = cols.join(',');
    const body = arr.map(o => cols.map(k=>{
      const v = (o[k]===undefined||o[k]===null)? '' : String(o[k]);
      return v.includes(',') ? '"' + v.replace(/"/g,'""') + '"' : v;
    }).join(',')).join('\n');
    return head + '\n' + body;
  }
};

class Preprocessor {
  constructor(model){
    this.impute = model.impute || null;
    this.normalize = model.normalize || null;
    this.pca = model.pca || {};
  }
  toBaseFeats(input){
    const feats = {};
    const baseNames = this.pca.featureNames || [];

    if(this.impute){
      const nums = Object.keys(this.impute.numeric||{});
      const facs = Object.keys(this.impute.factor||{});
      const levels = this.impute.categorical_levels || {};

      const raw = {};
      nums.forEach(k=>{
        const v = input[k];
        raw[k] = (v==='' || v==null)? null : parseFloat(v);
      });
      facs.forEach(k=>{
        const v = input[k];
        raw[k] = (v==='' || v==null)? null : String(v);
      });

      nums.forEach(k=>{ if(raw[k]===null || Number.isNaN(raw[k])) raw[k]=this.impute.numeric[k]; });
      facs.forEach(k=>{ if(raw[k]===null || raw[k]==='') raw[k]=this.impute.factor[k]; });

      baseNames.forEach(n=>feats[n]=0);
      nums.forEach(k=>{ if(k in feats) feats[k]=raw[k]; });
      facs.forEach(k=>{
        (levels[k]||[]).forEach(lv=>{
          const col = `${k}_${lv}`;
          if(col in feats) feats[col] = (raw[k]===lv)?1:0;
        });
      });
    }else{
      baseNames.forEach(k=>{
        const v = input[k];
        if(v===undefined || v==='') throw new Error(`Missing standardized feature: ${k}`);
        feats[k] = parseFloat(v);
      });
    }

    if(this.normalize && this.normalize.names){
      const names = this.normalize.names, means = this.normalize.means, sds = this.normalize.sds;
      for(let i=0;i<names.length;i++){
        const n = names[i];
        if(!(n in feats)) feats[n]=0;
        feats[n] = (feats[n] - means[i]) / (sds[i] || 1);
      }
    }
    return feats;
  }
}

class PCATransformer {
  constructor(pca){ this.pca = pca || {}; }
  project(feats){
    const featureNames = this.pca.featureNames || [];
    const rotation = this.pca.rotation || [];
    const pcNames = this.pca.pcNames || [];
    const cen = this.pca.center || null, scl = this.pca.scale || null;

    const x = featureNames.map(n => feats[n] ?? 0);
    const xcs = x.map((v,i)=> (scl ? ((cen? v-cen[i] : v)/(scl[i]||1)) : (cen? v-cen[i] : v)));

    const PCs = new Array(pcNames.length).fill(0);
    for(let k=0;k<pcNames.length;k++){
      let s=0; for(let j=0;j<xcs.length;j++) s += xcs[j] * rotation[j][k];
      PCs[k]=s;
    }
    return PCs;
  }
}

class RFPredictor {
  constructor(rf){ this.rf = rf || {}; }
  predictProbaPC(PCs){
    const trees = this.rf.trees || [];
    const K = (this.rf.classes||[]).length || (trees[0]?.[0]?.p?.length || 0);
    const acc = new Array(K).fill(0);
    for(const tree of trees){
      let node = tree[0];
      while(node && node.i !== undefined){
        const fi = node.i - 1;
        const thr = node.t;
        node = tree[((PCs[fi] <= thr ? node.l : node.r) - 1)];
      }
      const leaf = node && node.p ? node.p : new Array(K).fill(1/K);
      for(let k=0;k<K;k++) acc[k]+=leaf[k];
    }
    const denom = trees.length || 1;
    return acc.map(v => v/denom);
  }
}

class Calibrator {
  constructor(cals){ this.cals = cals || null; }
  platt(probs){
    if(!this.cals) return probs.slice();
    const out = [];
    for(let k=0;k<this.cals.length;k++){
      const ab = this.cals[k];
      const p = Math.min(Math.max(probs[k], 1e-6), 1-1e-6);
      const s = MathKit.logit(p);
      out.push(MathKit.sigmoid((ab?.[0]||0) + (ab?.[1]||1)*s));
    }
    return MathKit.rowNormalize(out);
  }
}

class CentroidSim {
  constructor(centroids){ this.M = (centroids && centroids.matrix) || []; }
  similar(PCs, alpha){
    const dists = this.M.map(row=>{
      let s=0; for(let i=0;i<row.length;i++){ const d=PCs[i]-row[i]; s+=d*d; }
      return Math.sqrt(s);
    });
    return MathKit.softmaxNegDist(dists, alpha);
  }
}

class Fusion {
  static fuse(Pcal, S, w){
    const K = Math.max(Pcal.length, S.length);
    const out = new Array(K);
    for(let k=0;k<K;k++) out[k] = (w*(Pcal[k]||0))+((1-w)*(S[k]||0));
    return out;
  }
}

/* =========================
   NEW: Input validation & constraints
   - Missing > 30% => reject ("Uncertain")
   - Binary numeric variables must be 0 or 1
   - Optional numeric bounds via model.meta.numericBounds
   ========================= */
const InputRules = {
  // 固定的二分类数值变量集合（按你的要求）：
  DEFAULT_BINARY_NUMERIC: new Set([
    'Female','Fever','Weight.loss','Proximal.pain','Peripheral.arthritis',
    'Headache','Jaw.claudication','Visual.symptoms','RF','CCP','ANA'
  ]),

  getFieldSets(model){
    if(model.impute){
      return {
        numKeys: Object.keys(model.impute.numeric||{}),
        facKeys: Object.keys(model.impute.factor||{}),
        catLevels: model.impute.categorical_levels || {}
      };
    }else{
      // 无插补信息时，回退到 PCA 的基础特征名
      const base = model.pca?.featureNames || [];
      return { numKeys: base, facKeys: [], catLevels: {} };
    }
  },

  getBinaryNumericSet(model){
    const list = model?.meta?.binaryNumeric || [];
    if(Array.isArray(list) && list.length) return new Set(list);
    return this.DEFAULT_BINARY_NUMERIC;
  },

  getNumericBounds(model){
    // 期望：model.meta.numericBounds = { Age:{min:0,max:120}, ESR:{min:0,max:200}, ... }
    return (model?.meta?.numericBounds) || {};
  },

  validateRecord(record, model){
    const {numKeys, facKeys, catLevels} = this.getFieldSets(model);
    const binSet = this.getBinaryNumericSet(model);
    const bounds = this.getNumericBounds(model);

    let missing = 0;
    const total = numKeys.length + facKeys.length;
    const problems = []; // e.g., BIN:<name>, RANGE:<name>, LEVEL:<name>, NaN:<name>

    for(const k of numKeys){
      let raw = record[k];
      if(raw === '' || raw === null || raw === undefined){ missing++; continue; }
      const v = (typeof raw === 'number') ? raw : parseFloat(raw);
      if(Number.isNaN(v)){ problems.push(`NaN:${k}`); continue; }
      if(binSet.has(k) && !(v === 0 || v === 1)){ problems.push(`BIN:${k}`); }
      const b = bounds[k];
      if(b && ((b.min!==undefined && v < b.min) || (b.max!==undefined && v > b.max))){
        problems.push(`RANGE:${k}`);
      }
    }

    for(const k of facKeys){
      const raw = record[k];
      if(raw === '' || raw === null || raw === undefined){ missing++; continue; }
      const levels = catLevels[k] || [];
      if(levels.length && !levels.includes(String(raw))){
        problems.push(`LEVEL:${k}`);
      }
    }

    const missRatio = total ? (missing/total) : 0;
    const ok = (problems.length === 0) && (missRatio <= 0.30);
    return { ok, missingRatio: missRatio, problems };
  },

  makeUncertainPred(model){
    const K = (model.meta?.classes || []).length;
    return {
      label: 'Uncertain',
      pmax: 0,
      Pfused: new Array(K).fill(0),
      S: new Array(K).fill(0),
      kmax: -1,
      PCs: []
    };
  }
};

class ModelRuntime {
  constructor(model){
    this.model = model;
    this.meta = model.meta || {};
    this.prep = new Preprocessor(model);
    this.pca = new PCATransformer(model.pca);
    this.rf = new RFPredictor(model.rf||{});
    this.cal = new Calibrator(model.calibrators||null);
    this.cent = new CentroidSim(model.centroids||{});
  }
  predictOne(record, override = {}){
    const w = override.w ?? this.meta.fusionWeight ?? 0.7;
    const alpha = override.alpha ?? this.meta.alphaDist ?? 1.0;
    const rej = override.rej ?? this.meta.rejectThreshold ?? 0.7; // default 0.70

    const feats = this.prep.toBaseFeats(record);
    const PCs  = this.pca.project(feats);
    const Praw = this.rf.predictProbaPC(PCs);
    const Pcal = this.cal.platt(Praw);
    const S    = this.cent.similar(PCs, alpha);
    const Pfused = Fusion.fuse(Pcal, S, w);

    const pmax = Math.max(...Pfused);
    const kmax = MathKit.argmax(Pfused);
    const classes = this.meta.classes || [];
    const label = (pmax >= rej)? (classes[kmax] ?? String(kmax)) : 'Uncertain';

    return { label, pmax, Pfused, S, kmax, PCs };
  }
  makeTemplateColumns(){
    if(this.model.impute){
      const nums = Object.keys(this.model.impute.numeric||{});
      const facs = Object.keys(this.model.impute.factor||{});
      return [...nums, ...facs];
    }
    return this.model.pca?.featureNames || [];
  }
}

/* =========================
   View Layer (UI)
   ========================= */
const View = {
  els: {
    modelMeta: document.getElementById('modelMeta'),
    modelUrl:  document.getElementById('modelUrl'),
    loadBtn:   document.getElementById('loadBtn'),
    tplBtn:    document.getElementById('tplBtn'),
    loadState: document.getElementById('loadState'),

    wInput:    document.getElementById('wInput'),
    alphaInput:document.getElementById('alphaInput'),
    rejInput:  document.getElementById('rejInput'),

    inputForm: document.getElementById('inputForm'),
    predictBtn:document.getElementById('predictBtn'),
    resetBtn:  document.getElementById('resetBtn'),
    predState: document.getElementById('predState'),

    csvFile:   document.getElementById('csvFile'),
    batchBtn:  document.getElementById('batchBtn'),
    dlLink:    document.getElementById('dlLink'),
    batchState:document.getElementById('batchState'),

    result:    document.getElementById('result')
  },

  setLoadState(msg){ if (this.els.loadState) this.els.loadState.textContent = msg; },
  setPredState(msg){ if (this.els.predState) this.els.predState.textContent = msg; },
  setBatchState(msg){ if (this.els.batchState) this.els.batchState.textContent = msg; },

  enableAfterLoad(){
    if (this.els.predictBtn) this.els.predictBtn.disabled=false;
    if (this.els.resetBtn)   this.els.resetBtn.disabled=false;
    if (this.els.batchBtn)   this.els.batchBtn.disabled=false;
    if (this.els.tplBtn)     this.els.tplBtn.disabled=false;
  },

  renderMeta(model){
    if (!this.els.modelMeta) return;
    const meta = model.meta || {};
    const cls = (meta.classes || model.rf?.classes || []).join(', ');
    const hasImpute = !!model.impute;
    const hasNorm = !!model.normalize;
    this.els.modelMeta.innerHTML =
      `Classes: <b>${cls}</b> | #PCs: <b>${(model.pca?.pcNames||[]).length}</b> | Trees: <b>${model.rf?.trees?.length||0}</b><br>` +
      `Preprocessing: ${hasImpute?'Imputation ✓':'Imputation ×'}; ${hasNorm?'Standardization ✓':'Standardization ×'}; PCA ✓`;
  },

  renderForm(model){
    const wrap = this.els.inputForm;
    if (!wrap) return;
    wrap.innerHTML = '';

    // 获取限制集合
    const binSet = InputRules.getBinaryNumericSet(model);
    const numBounds = InputRules.getNumericBounds(model);

    const fieldNumber = (name)=>{
      const inp = document.createElement('input');
      inp.type='number'; inp.step='any'; inp.id='fld_'+name;

      // 二分类数值强制 0/1
      if(binSet.has(name)){
        inp.min = 0; inp.max = 1; inp.step = 1; inp.placeholder = '0 or 1';
      }
      // 数值范围限制（若在 model.meta.numericBounds 提供）
      const b = numBounds[name];
      if(b){
        if(b.min !== undefined) inp.min = b.min;
        if(b.max !== undefined) inp.max = b.max;
      }
      return this._containerFor(name, inp);
    };

    if(model.impute){
      const numKeys = Object.keys(model.impute.numeric||{});
      const facKeys = Object.keys(model.impute.factor||{});
      const catLevels = model.impute.categorical_levels || {};

      numKeys.forEach(k=>wrap.appendChild(fieldNumber(k)));

      facKeys.forEach(k=>{
        const sel = document.createElement('select'); sel.id = 'fld_'+k;
        const opt0 = document.createElement('option'); opt0.value=''; opt0.textContent='(missing)'; sel.appendChild(opt0);
        (catLevels[k]||[]).forEach(v=>{
          const o = document.createElement('option'); o.value=v; o.textContent=v; sel.appendChild(o);
        });
        wrap.appendChild(this._containerFor(k, sel));
      });
    }else{
      const baseNames = model.pca?.featureNames || [];
      baseNames.forEach(k=>wrap.appendChild(fieldNumber(k)));
      const warn = document.createElement('div');
      warn.className='hint warn';
      warn.textContent = 'Model lacks imputation/standardization params: please provide pre-imputed & standardized values aligned with featureNames.';
      wrap.appendChild(warn);
    }
  },

  _containerFor(name, node){
    const div = document.createElement('div'); div.className='col';
    const lab = document.createElement('label'); lab.htmlFor='fld_'+name; lab.textContent=name;
    div.appendChild(lab); div.appendChild(node);
    return div;
  },

  readSingleInput(model){
    const record = {};
    if (!this.els.inputForm) return record;
    if(model.impute){
      const numKeys = Object.keys(model.impute.numeric||{});
      const facKeys = Object.keys(model.impute.factor||{});
      numKeys.forEach(k=>{
        const v = document.getElementById('fld_'+k)?.value ?? '';
        record[k] = (v===''? null : parseFloat(v));
      });
      facKeys.forEach(k=>{
        const v = document.getElementById('fld_'+k)?.value ?? '';
        record[k] = (v===''? null : v);
      });
    }else{
      const baseNames = model.pca?.featureNames || [];
      baseNames.forEach(k=>{
        const v = document.getElementById('fld_'+k)?.value ?? '';
        record[k] = v;
      });
    }
    return record;
  },

  clearForm(){
    this.setPredState('');
    if (this.els.result) this.els.result.textContent='No results yet';
    if (!this.els.inputForm) return;
    this.els.inputForm.querySelectorAll('input,select').forEach(el=>el.value='');
  },

  renderResult(model, pred, ms, extraNote){
    if (!this.els.result) return;
    const note = extraNote ? `<div class="hint warn" style="margin:6px 0">${extraNote}</div>` : '';
    const cls = model.meta.classes || [];
    const rows = pred.Pfused.map((p,i)=>`<tr><td>${cls[i]||i}</td><td>${p.toFixed(6)}</td><td>${pred.S[i].toFixed(6)}</td></tr>`).join('');
    this.els.result.innerHTML =
      `${note}<p>Predicted subtype: <b>${pred.label}</b> | Confidence p<sub>max</sub>=<b>${pred.pmax.toFixed(6)}</b> | Time ${ms.toFixed(1)} ms</p>
       <table><thead><tr><th>Class</th><th>Fused probability</th><th>Centroid similarity</th></tr></thead><tbody>${rows}</tbody></table>`;
  },

  downloadBlob(filename, blob){
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename; a.click();
    URL.revokeObjectURL(url);
  },

  showDownloadLink(blob){
    if (!this.els.dlLink) return;
    const url = URL.createObjectURL(blob);
    this.els.dlLink.href = url;
    this.els.dlLink.style.display='inline-block';
  }
};

/* =========================
   Controller (events)
   ========================= */
const Controller = {
  runtime: null,

  async onLoadModel(){
    View.setLoadState('Loading…');
    const url = (View.els.modelUrl?.value || 'model.json').trim();
    try{
      const res = await fetch(url, {cache:'no-store'});
      if(!res.ok) throw new Error(`HTTP ${res.status}`);
      const model = await res.json();

      const meta = model.meta || {};
      if (View.els.wInput)     View.els.wInput.value     = meta.fusionWeight ?? 0.7;
      if (View.els.alphaInput) View.els.alphaInput.value = meta.alphaDist ?? 1.0;
      if (View.els.rejInput)   View.els.rejInput.value   = meta.rejectThreshold ?? 0.7;

      this.runtime = new ModelRuntime(model);
      View.renderForm(model);
      View.renderMeta(model);
      View.enableAfterLoad();
      View.setLoadState('Loaded');
    }catch(e){
      console.error(e);
      View.setLoadState('Load failed: ' + e.message);
    }
  },

  onDownloadTemplate(){
    if(!this.runtime) return;
    const cols = this.runtime.makeTemplateColumns();
    const header = cols.join(',');
    const blob = new Blob([header+'\n'], {type:'text/csv;charset=utf-8;'});
    View.downloadBlob('template.csv', blob);
  },

  onReset(){
    View.clearForm();
  },

  onPredictOne(){
    if(!this.runtime){ alert('Please load the model first'); return; }
    const t0 = performance.now();
    const model = this.runtime.model;

    const w = parseFloat(View.els.wInput?.value || (model.meta?.fusionWeight ?? 0.7));
    const alpha = parseFloat(View.els.alphaInput?.value || (model.meta?.alphaDist ?? 1.0));
    const rej = parseFloat(View.els.rejInput?.value || (model.meta?.rejectThreshold ?? 0.7));

    try{
      const record = View.readSingleInput(model);

      // --- 校验（缺失>30%=>Uncertain；二分类与范围限制） ---
      const v = InputRules.validateRecord(record, model);
      if(v.missingRatio > 0.30){
        const pred = InputRules.makeUncertainPred(model);
        View.setPredState('Auto-rejected: missing rate > 30%');
        View.renderResult(model, pred, performance.now()-t0, 'Auto-rejected due to missing > 30%.');
        return;
      }
      if(!v.ok){
        View.setPredState('Invalid input: ' + v.problems.join('; '));
        return;
      }
      // -----------------------------------------------------

      const pred = this.runtime.predictOne(record, {w, alpha, rej});
      View.setPredState('Done');
      View.renderResult(model, pred, performance.now()-t0);
    }catch(e){
      console.error(e);
      View.setPredState('Failed: ' + e.message);
    }
  },

  async onBatchPredict(){
    if(!this.runtime){ alert('Please load the model first'); return; }
    const file = View.els.csvFile?.files?.[0];
    if(!file){ alert('Please choose a CSV file'); return; }

    const model = this.runtime.model;
    const w = parseFloat(View.els.wInput?.value || (model.meta?.fusionWeight ?? 0.7));
    const alpha = parseFloat(View.els.alphaInput?.value || (model.meta?.alphaDist ?? 1.0));
    const rej = parseFloat(View.els.rejInput?.value || (model.meta?.rejectThreshold ?? 0.7));

    try{
      View.setBatchState('Parsing…');
      const text = await file.text();
      const {rows} = CSVKit.parse(text);
      if(!rows.length) throw new Error('Empty file');

      View.setBatchState(`Total ${rows.length}, predicting…`);
      const results = [];
      const classes = model.meta?.classes || [];
      for(let i=0;i<rows.length;i++){
        const rec = rows[i];

        // --- 每行校验 ---
        const v = InputRules.validateRecord(rec, model);
        let qc_flag = '';
        if(v.missingRatio > 0.30) qc_flag = 'MISSING>30%';
        if(v.problems.length){ qc_flag = (qc_flag? qc_flag+'|' : '') + v.problems.join('|'); }

        if(qc_flag){
          const out = { pred_cluster: 'Uncertain', p_max: 0, qc_flag };
          classes.forEach((c,idx)=>{ out['P_'+c] = 0; });
          results.push({...rec, ...out});
        }else{
          const pred = this.runtime.predictOne(rec, {w, alpha, rej});
          const out = { pred_cluster: pred.label, p_max: +pred.pmax.toFixed(6), qc_flag };
          classes.forEach((c,idx)=>{ out['P_'+c] = +pred.Pfused[idx].toFixed(6); });
          results.push({...rec, ...out});
        }
        // ----------------

        if((i+1)%50===0) View.setBatchState(`Processed ${i+1}/${rows.length}…`);
      }
      View.setBatchState(`Done ${rows.length}`);
      const csv = CSVKit.toCSV(results);
      const blob = new Blob([csv], {type:'text/csv;charset=utf-8;'});
      View.showDownloadLink(blob);
    }catch(e){
      console.error(e);
      View.setBatchState('Failed: ' + e.message);
    }
  },

  bind(){
    View.els.loadBtn?.addEventListener('click', ()=>this.onLoadModel());
    View.els.tplBtn?.addEventListener('click', ()=>this.onDownloadTemplate());
    View.els.resetBtn?.addEventListener('click', ()=>this.onReset());
    View.els.predictBtn?.addEventListener('click', ()=>this.onPredictOne());
    View.els.batchBtn?.addEventListener('click', ()=>this.onBatchPredict());
  }
};

Controller.bind();
