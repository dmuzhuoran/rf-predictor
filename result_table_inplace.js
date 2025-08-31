
/**
 * result_table_inplace_v2.js
 * Robust in-place renderer:
 *  - Debounced MutationObserver (handles rapid rewrites).
 *  - Parses from element.innerText (tolerates <br/> etc.).
 *  - Replaces ONLY the "Class Fused probability Centroid similarity" block
 *    with a 3-column table at the SAME spot, keeping the lines above intact.
 *  - Works on every new prediction.
 *  - Exposes window.__fillTable(fused, similarity) to force a render in place.
 */
(function(){
  'use strict';

  const box = document.getElementById('result');
  if (!box) return;

  // tolerant header & numeric row detection
  const HEADER_RE = /class\s+fused\s+probability\s+centroid\s+similarity/i;
  const NUM_SRC = String.raw`[+-]?(?:\\d+(?:\\.\\d+)?|\\.\\d+)(?:e[+-]?\\d+)?`;
  const ROW_RE = new RegExp(String.raw`^\\s*(\\d+)\\s+(${NUM_SRC})\\s+(${NUM_SRC})\\s*$`, 'i');

  const to6 = (x) => Number.isFinite(x) ? x.toFixed(6) : "0.000000";
  const esc = (s) => s.replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));

  function getLines() {
    // use innerText so line breaks from <br> are preserved
    let txt = (box.innerText ?? box.textContent ?? "");
    // normalize NBSP and multiple spaces to single spaces for robust matching
    txt = txt.replace(/\u00A0/g, ' ');
    return txt.split(/\r?\n/);
  }

  function parse() {
    const lines = getLines();
    const idx = lines.findIndex(line => HEADER_RE.test(line));
    if (idx < 0) return null;

    const fused=[], sim=[];
    for (let i=1;i<=3;i++) {
      const raw = (lines[idx+i] || "").replace(/\s{2,}/g, ' ').trim();
      const m = raw.match(ROW_RE);
      if (m) {
        fused.push(parseFloat(m[2]));
        sim.push(parseFloat(m[3]));
      }
    }
    const prefix = lines.slice(0, idx).join('\n').trimEnd();
    return { prefix, fused, sim };
  }

  function buildTableHTML(fused, sim) {
    const K = Math.max(3, fused.length, sim.length);
    let rows = "";
    for (let i=0;i<K;i++) {
      rows += `<tr>
        <td>${i+1}</td>
        <td>${to6(fused[i])}</td>
        <td>${to6(sim[i])}</td>
      </tr>`;
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

  function renderInPlaceFromCurrentText() {
    const parsed = parse();
    if (!parsed || !parsed.fused.length) return false;

    const prefixHTML = parsed.prefix
      ? `<div class="mono">${esc(parsed.prefix).replace(/\n/g, "<br>")}</div>`
      : "";
    const tableHTML = buildTableHTML(parsed.fused, parsed.sim);
    box.innerHTML = prefixHTML + tableHTML;
    return true;
  }

  // Debounced observer so our render wins after the app finishes writing
  let t = null;
  const schedule = () => {
    clearTimeout(t);
    t = setTimeout(renderInPlaceFromCurrentText, 80); // 80ms debounce
  };

  // Initial attempt (in case results already present)
  renderInPlaceFromCurrentText();

  const obs = new MutationObserver(schedule);
  obs.observe(box, { childList:true, subtree:true, characterData:true });

  // Manual API: write at the same location
  window.__fillTable = function(fused=[], similarity=[]) {
    // Use current prefix if available, otherwise keep existing content above
    const lines = getLines();
    const idx = lines.findIndex(line => HEADER_RE.test(line));
    const prefix = (idx >= 0 ? lines.slice(0, idx).join('\n') : lines.join('\n')).trimEnd();
    const prefixHTML = prefix ? `<div class="mono">${esc(prefix).replace(/\n/g, "<br>")}</div>` : "";
    box.innerHTML = prefixHTML + buildTableHTML(fused, similarity);
  };
})();
