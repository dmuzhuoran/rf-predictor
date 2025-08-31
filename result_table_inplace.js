
/**
 * result_table_inplace.js
 * Render the 3-class table exactly where the current text shows the
 * "Class Fused probability Centroid similarity" section.
 * - Keeps the lines above (e.g., auto-reject, predicted subtype).
 * - Replaces only the "Class …" 3-line block with a styled table.
 * - Watches #result for updates; works with subsequent predictions.
 * - No changes to your existing app logic.
 */
(function(){
  'use strict';

  const resultBox = document.getElementById('result');
  if(!resultBox) return;

  // regex to capture numeric rows (supports ints, decimals, scientific)
  const NUM = String.raw`[+-]?(?:\\d+(?:\\.\\d+)?|\\.\\d+)(?:e[+-]?\\d+)?`;
  const rowRe = new RegExp(String.raw`^(\\d+)\\s+(${NUM})\\s+(${NUM})$`, "i");
  const headerRe = /^Class\s+Fused\s+probability\s+Centroid\s+similarity/i;

  function esc(s){
    return s.replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
  }

  function buildTableHTML(fused, sim){
    const to6 = x => Number.isFinite(x) ? x.toFixed(6) : "0.000000";
    let rows = "";
    const K = Math.max(3, fused.length, sim.length);
    for(let i=0;i<K;i++){
      const c = i+1, p = fused[i], s = sim[i];
      rows += `<tr><td>${c}</td><td>${to6(p)}</td><td>${to6(s)}</td></tr>`;
    }
    return `
<table class="result-table" style="border-collapse:collapse;margin-top:10px;width:100%">
  <thead>
    <tr>
      <th style="text-align:center;padding:6px 18px;border-bottom:1px solid var(--bd);background:var(--accent-weak)">Class</th>
      <th style="text-align:center;padding:6px 18px;border-bottom:1px solid var(--bd);background:var(--accent-weak)">Fused Probability</th>
      <th style="text-align:center;padding:6px 18px;border-bottom:1px solid var(--bd);background:var(--accent-weak)">Centroid Similarity</th>
    </tr>
  </thead>
  <tbody id="resultBody">
    ${rows}
  </tbody>
</table>`;
  }

  function renderInPlaceFromText(txt){
    if(!txt) return false;
    const lines = txt.split(/\r?\n/);

    // find the "Class …" header
    let start = -1;
    for(let i=0;i<lines.length;i++){
      if(headerRe.test(lines[i].trim())) { start = i; break; }
    }
    if(start < 0) return false;

    // parse 3 rows after header
    let fused=[], sim=[];
    for(let i=1;i<=3;i++){
      const m = (lines[start+i]||"").trim().match(rowRe);
      if(m){
        fused.push(parseFloat(m[2]));
        sim.push(parseFloat(m[3]));
      }
    }

    // prefix text (everything before the header) we keep as-is (with <br>)
    const prefix = lines.slice(0, start).join("\n").trimEnd();

    // build final HTML: prefix lines (preserved) + table
    const prefixHTML = prefix ? `<div class="mono">${esc(prefix).replace(/\n/g,"<br>")}</div>` : "";
    const tableHTML  = buildTableHTML(fused, sim);

    // Write back (replace only the section starting at header)
    // Simpler strategy: replace entire innerHTML with prefix + table.
    // Your app usually sets resultBox.innerText per prediction, so we'll
    // watch and re-run for new predictions.
    resultBox.innerHTML = prefixHTML + tableHTML;
    return true;
  }

  // Try once immediately
  renderInPlaceFromText(resultBox.textContent || "");

  // Watch for prediction updates
  const obs = new MutationObserver(() => {
    // If new text still has the "Class …" header, re-render into table
    // If result already contains our table but no text header, do nothing
    // (it means we've already converted).
    const txt = resultBox.textContent || "";
    if (headerRe.test(txt)) renderInPlaceFromText(txt);
  });
  obs.observe(resultBox, { childList:true, subtree:true, characterData:true });

  // Optional: manual API (also renders at the same location below the prefix)
  window.__fillTable = function(fused=[], similarity=[]){
    // Build a minimal prefix using current text up to (but not including) header
    const txt = resultBox.textContent || "";
    const lines = txt.split(/\r?\n/);
    let start = -1;
    for(let i=0;i<lines.length;i++){
      if(headerRe.test(lines[i].trim())) { start = i; break; }
    }
    const prefix = start>=0 ? lines.slice(0,start).join("\n").trimEnd() : txt.trimEnd();
    const prefixHTML = prefix ? `<div class="mono">${esc(prefix).replace(/\n/g,"<br>")}</div>` : "";
    resultBox.innerHTML = prefixHTML + buildTableHTML(fused, similarity);
  };
})();
