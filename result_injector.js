
// result_injector.js — non-invasive table filler
// Works even if internal View.renderResult is unknown.
// Strategy: 1) expose a global helper __fillTable; 2) observe #result text changes;
// 3) try to parse the monospace text block to extract three rows; 4) handle auto-reject message.

(function(){
  const tbody = document.getElementById('resultBody');
  const resultBox = document.getElementById('result');
  if(!tbody || !resultBox) return;

  function clearNote(note = "No results yet"){
    tbody.innerHTML = `<tr><td colspan="3" style="text-align:center;color:var(--muted)">${note}</td></tr>`;
  }

  function fillTable(classes, fused, sim){
    tbody.innerHTML = "";
    const K = Math.max(classes.length, fused.length, sim.length, 3);
    for(let i=0;i<K;i++){
      const c = classes[i] ?? (i+1);
      const p = Number.isFinite(fused[i]) ? fused[i] : 0;
      const s = Number.isFinite(sim[i]) ? sim[i] : 0;
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${c}</td><td>${p.toFixed(6)}</td><td>${s.toFixed(6)}</td>`;
      tbody.appendChild(tr);
    }
  }

  // expose a global helper for optional direct calls
  window.__fillTable = (fused=[], sim=[]) => fillTable([1,2,3], fused, sim);

  function parseTextAndFill(txt){
    // Auto-reject?
    if(/Auto-?rejected/i.test(txt)){
      clearNote("Auto-rejected: missing rate > 30%");
      return true;
    }
    // Look for section starting with "Class" header and then lines beginning with 1/2/3
    const lines = txt.split(/\r?\n/).map(s => s.trim()).filter(Boolean);
    const startIdx = lines.findIndex(s => /^Class\b/i.test(s));
    if(startIdx >= 0){
      let fused=[], sim=[];
      for(let i=1;i<=3;i++){
        const l = lines[startIdx + i];
        if(!l) continue;
        // expected pattern: "<class> <p> <s>"
        const m = l.match(/^(\d+)\s+([0-9.]+)\s+([0-9.]+)$/);
        if(m){
          fused.push(parseFloat(m[2]));
          sim.push(parseFloat(m[3]));
        }
      }
      if(fused.length){
        fillTable([1,2,3], fused, sim);
        return true;
      }
    }
    return false;
  }

  // Initial try (in case results already present)
  parseTextAndFill(resultBox.textContent || "");

  // Observe future changes
  const obs = new MutationObserver(() => {
    const ok = parseTextAndFill(resultBox.textContent || "");
    // don't disconnect; allow multiple predictions
  });
  obs.observe(resultBox, { childList: true, subtree: true, characterData: true });
})();
