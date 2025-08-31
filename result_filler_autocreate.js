
/**
 * result_filler_autocreate.js
 * Drop-in, zero-config table writer.
 * - If #resultBody doesn't exist, it will CREATE the whole table inside #result.
 * - Then same behavior as result_filler.js (wrap View.renderResult + observe text).
 * - Exposes window.__fillTable(fused, similarity, classes?).
 */
(function(){
  'use strict';

  const resultBox = document.getElementById('result');
  if(!resultBox) return;

  // Ensure table exists
  let tbody = document.getElementById('resultBody');
  if(!tbody){
    const table = document.createElement('table');
    table.className = 'result-table';
    table.style.borderCollapse = 'collapse';
    table.style.marginTop = '10px';
    table.style.width = '100%';

    const thead = document.createElement('thead');
    thead.innerHTML = '<tr><th>Class</th><th>Fused Probability</th><th>Centroid Similarity</th></tr>';
    tbody = document.createElement('tbody');
    tbody.id = 'resultBody';
    tbody.innerHTML = '<tr><td colspan="3" style="text-align:center;color:var(--muted)">No results yet</td></tr>';

    table.appendChild(thead);
    table.appendChild(tbody);
    resultBox.appendChild(table);
  }

  function clearNote(note = "No results yet") {
    tbody.innerHTML = `<tr><td colspan="3" style="text-align:center;color:var(--muted)">${note}</td></tr>`;
  }
  function to6(x){return Number.isFinite(x)?x.toFixed(6):"0.000000";}
  function fillTable(classes=[1,2,3], fused=[], sim=[]){
    tbody.innerHTML = "";
    const K = Math.max(classes.length, fused.length, sim.length, 3);
    for(let i=0;i<K;i++){
      const c = classes[i] ?? (i+1);
      const p = fused[i], s = sim[i];
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${c}</td><td>${to6(p)}</td><td>${to6(s)}</td>`;
      tr.style.textAlign = 'center';
      // add some spacing
      Array.from(tr.children).forEach(td => { td.style.padding = '6px 18px'; td.style.borderBottom = '1px solid var(--bd)'; });
      tbody.appendChild(tr);
    }
  }
  window.__fillTable = function (fused=[], similarity=[], classes=[1,2,3]) {
    fillTable(classes, fused, similarity);
  };

  // Try to wrap View.renderResult if present
  try{
    if (window.View && typeof window.View.renderResult === "function") {
      const _orig = window.View.renderResult.bind(window.View);
      window.View.renderResult = function (model, pred, ms, extraNote) {
        const ret = _orig(model, pred, ms, extraNote);
        try{
          const classes = (model && model.meta && Array.isArray(model.meta.classes))
            ? model.meta.classes : [1,2,3];
          const fused = (pred && Array.isArray(pred.Pfused)) ? pred.Pfused : [];
          const sim   = (pred && Array.isArray(pred.S))      ? pred.S      : [];
          const uncertain = (pred && pred.label === 'Uncertain');
          const allZeros  = (!fused.length || fused.every(v=>!v)) &&
                            (!sim.length   || sim.every(v=>!v));
          const hasReject = typeof extraNote==='string' && /Auto-?rejected/i.test(extraNote);
          if (hasReject || (uncertain && allZeros)) clearNote("Auto-rejected: missing rate > 30%");
          else if (fused.length || sim.length) fillTable(classes, fused, sim);
        }catch(e){ console.error("result_filler_autocreate wrapper error:", e); }
        return ret;
      };
    }
  }catch(e){ console.warn("result_filler_autocreate: cannot wrap View.renderResult:", e); }

  // Fallback: parse text in #result
  const NUM = String.raw`[+-]?(?:\\d+(?:\\.\\d+)?|\\.\\d+)(?:e[+-]?\\d+)?`;
  const rowRe = new RegExp(String.raw`^(\\d+)\\s+(${NUM})\\s+(${NUM})$`, "i");
  function parseTextAndFill(txt){
    if(/Auto-?rejected/i.test(txt)){ clearNote("Auto-rejected: missing rate > 30%"); return true; }
    const lines = (txt||"").split(/\r?\n/).map(s=>s.trim());
    const headerIdx = lines.findIndex(s=>/^Class\b/i.test(s));
    if (headerIdx < 0) return false;
    const fused=[], sim=[];
    for(let i=1;i<=3;i++){
      const m = (lines[headerIdx+i]||"").match(rowRe);
      if(m){ fused.push(parseFloat(m[2])); sim.push(parseFloat(m[3])); }
    }
    if(fused.length){ fillTable([1,2,3], fused, sim); return true; }
    return false;
  }
  // Try immediately + observe
  parseTextAndFill(resultBox.textContent || "");
  const obs = new MutationObserver(()=>{ parseTextAndFill(resultBox.textContent || ""); });
  obs.observe(resultBox, { childList:true, subtree:true, characterData:true });
})();
