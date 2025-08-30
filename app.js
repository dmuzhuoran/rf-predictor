/* =========================
   功能模块（Core Layer）
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
      const cols = lines[i].split(','); // 简化：不处理复杂转义
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
        if(v===undefined || v==='') throw new Error(`缺少已标准化特征：${k}`);
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
    const rej = override.rej ?? this.meta.rejectThreshold ?? 0.7;

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
   展示模块（UI Layer）
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

  setLoadState(msg){ this.els.loadState.textContent = msg; },
  setPredState(msg){ this.els.predState.textContent = msg; },
  setBatchState(msg){ this.els.batchState.textContent = msg; },

  enableAfterLoad(){
    this.els.predictBtn.disabled=false;
    this.els.resetBtn.disabled=false;
    this.els.batchBtn.disabled=false;
    this.els.tplBtn.disabled=false;
  },

  renderMeta(model){
    const meta = model.meta || {};
    const cls = (meta.classes || model.rf?.classes || []).join(', ');
    const hasImpute = !!model.impute;
    const hasNorm = !!model.normalize;
    this.els.modelMeta.innerHTML =
      `类别：<b>${cls}</b> ｜ 主成分数：<b>${(model.pca?.pcNames||[]).length}</b> ｜ 树数：<b>${model.rf?.trees?.length||0}</b><br>` +
      `预处理：${hasImpute?'插补✓':'插补×'}；${hasNorm?'标准化✓':'标准化×'}；PCA✓`;
  },

  renderForm(model){
    const wrap = this.els.inputForm;
    wrap.innerHTML = '';

    const fieldNumber = (name)=>{
      const inp = document.createElement('input');
      inp.type='number'; inp.step='any'; inp.id='fld_'+name;
      return this._containerFor(name, inp);
    };

    if(model.impute){
      const numKeys = Object.keys(model.impute.numeric||{});
      const facKeys = Object.keys(model.impute.factor||{});
      const catLevels = model.impute.categorical_levels || {};

      numKeys.forEach(k=>wrap.appendChild(fieldNumber(k)));

      facKeys.forEach(k=>{
        const sel = document.createElement('select'); sel.id = 'fld_'+k;
        const opt0 = document.createElement('option'); opt0.value=''; opt0.textContent='(缺失)'; sel.appendChild(opt0);
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
      warn.textContent = '当前模型不含插补/标准化参数：请提供“已插补 + 已标准化”的数值，并与 featureNames 完全对齐。';
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
    this.els.result.textContent='尚无结果';
    this.els.inputForm.querySelectorAll('input,select').forEach(el=>el.value='');
  },

  renderResult(model, pred, ms){
    const cls = model.meta.classes || [];
    const rows = pred.Pfused.map((p,i)=>`<tr><td>${cls[i]||i}</td><td>${p.toFixed(6)}</td><td>${pred.S[i].toFixed(6)}</td></tr>`).join('');
    this.els.result.innerHTML =
      `<p>预测分型：<b>${pred.label}</b> ｜ 置信度 p<sub>max</sub>=<b>${pred.pmax.toFixed(6)}</b> ｜ 用时 ${ms.toFixed(1)} ms</p>
       <table><thead><tr><th>分型</th><th>融合后概率</th><th>中心相似度</th></tr></thead><tbody>${rows}</tbody></table>`;
  },

  downloadBlob(filename, blob){
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename; a.click();
    URL.revokeObjectURL(url);
  },

  showDownloadLink(blob){
    const url = URL.createObjectURL(blob);
    this.els.dlLink.href = url;
    this.els.dlLink.style.display='inline-block';
  }
};

/* =========================
   控制器（事件与协调）
   ========================= */
const Controller = {
  runtime: null,

  async onLoadModel(){
    View.setLoadState('加载中…');
    const url = (View.els.modelUrl.value || 'model.json').trim();
    try{
      const res = await fetch(url, {cache:'no-store'});
      if(!res.ok) throw new Error(`HTTP ${res.status}`);
      const model = await res.json();

      const meta = model.meta || {};
      View.els.wInput.value     = meta.fusionWeight ?? 0.7;
      View.els.alphaInput.value = meta.alphaDist ?? 1.0;
      View.els.rejInput.value   = meta.rejectThreshold ?? 0.7;

      this.runtime = new ModelRuntime(model);
      View.renderForm(model);
      View.renderMeta(model);
      View.enableAfterLoad();
      View.setLoadState('已加载');
    }catch(e){
      console.error(e);
      View.setLoadState('加载失败：' + e.message);
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
    if(!this.runtime){ alert('请先加载模型'); return; }
    const t0 = performance.now();
    const model = this.runtime.model;

    const w = parseFloat(View.els.wInput.value || (model.meta?.fusionWeight ?? 0.7));
    const alpha = parseFloat(View.els.alphaInput.value || (model.meta?.alphaDist ?? 1.0));
    const rej = parseFloat(View.els.rejInput.value || (model.meta?.rejectThreshold ?? 0.7));

    try{
      const record = View.readSingleInput(model);
      const pred = this.runtime.predictOne(record, {w, alpha, rej});
      View.setPredState('完成');
      View.renderResult(model, pred, performance.now()-t0);
    }catch(e){
      console.error(e);
      View.setPredState('失败：' + e.message);
    }
  },

  async onBatchPredict(){
    if(!this.runtime){ alert('请先加载模型'); return; }
    const file = View.els.csvFile.files[0];
    if(!file){ alert('请选择 CSV 文件'); return; }

    const model = this.runtime.model;
    const w = parseFloat(View.els.wInput.value || (model.meta?.fusionWeight ?? 0.7));
    const alpha = parseFloat(View.els.alphaInput.value || (model.meta?.alphaDist ?? 1.0));
    const rej = parseFloat(View.els.rejInput.value || (model.meta?.rejectThreshold ?? 0.7));

    try{
      View.setBatchState('解析中…');
      const text = await file.text();
      const {rows} = CSVKit.parse(text);
      if(!rows.length) throw new Error('空文件');

      View.setBatchState(`共 ${rows.length} 条，预测中…`);
      const results = [];
      const classes = model.meta?.classes || [];
      for(let i=0;i<rows.length;i++){
        const rec = rows[i];
        const pred = this.runtime.predictOne(rec, {w, alpha, rej});
        const out = { pred_cluster: pred.label, p_max: +pred.pmax.toFixed(6) };
        classes.forEach((c,idx)=>{ out['P_'+c] = +pred.Pfused[idx].toFixed(6); });
        results.push({...rec, ...out});
        if((i+1)%50===0) View.setBatchState(`已完成 ${i+1}/${rows.length}…`);
      }
      View.setBatchState(`完成 ${rows.length} 条`);
      const csv = CSVKit.toCSV(results);
      const blob = new Blob([csv], {type:'text/csv;charset=utf-8;'});
      View.showDownloadLink(blob);
    }catch(e){
      console.error(e);
      View.setBatchState('失败：' + e.message);
    }
  },

  bind(){
    View.els.loadBtn.addEventListener('click', ()=>this.onLoadModel());
    View.els.tplBtn.addEventListener('click', ()=>this.onDownloadTemplate());
    View.els.resetBtn.addEventListener('click', ()=>this.onReset());
    View.els.predictBtn.addEventListener('click', ()=>this.onPredictOne());
    View.els.batchBtn.addEventListener('click', ()=>this.onBatchPredict());
  }
};

Controller.bind();

