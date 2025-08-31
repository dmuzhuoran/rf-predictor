
/**
 * result_filler.js
 * Non-invasive result table writer for the GPSD predictor page.
 * - If window.View.renderResult exists, we wrap it to also write the table.
 * - Otherwise we watch #result for text changes and parse the monospace block.
 * - Exposes window.__fillTable(fused, similarity, classes?) for manual use.
 *
 * Drop this file right before </body>:
 *   <script src="./result_filler.js"></script>
 */
(function () {
  'use strict';

  // ---------- DOM helpers ----------
  const resultBox = document.getElementById('result');
  const tbody = document.getElementById('resultBody');

  if (!resultBox || !tbody) {
    // Silently do nothing if the expected nodes are not present.
    // (Keeps this script non-breaking on other pages.)
    return;
  }

  function clearNote(note = "No results yet") {
    tbody.innerHTML =
      `<tr><td colspan="3" style="text-align:center;color:var(--muted)">${note}</td></tr>`;
  }

  function to6(x) {
    return Number.isFinite(x) ? x.toFixed(6) : "0.000000";
  }

  function fillTable(classes = [1, 2, 3], fused = [], sim = []) {
    tbody.innerHTML = "";
    const K = Math.max(classes.length, fused.length, sim.length, 3);
    for (let i = 0; i < K; i++) {
      const c = classes[i] ?? (i + 1);
      const p = fused[i];
      const s = sim[i];
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${c}</td><td>${to6(p)}</td><td>${to6(s)}</td>`;
      tbody.appendChild(tr);
    }
  }

  // Expose manual helper so your own code can push results directly.
  window.__fillTable = function (fused = [], similarity = [], classes = [1, 2, 3]) {
    fillTable(classes, fused, similarity);
  };

  // ---------- Optional: wrap View.renderResult if present ----------
  try {
    if (window.View && typeof window.View.renderResult === "function") {
      const _orig = window.View.renderResult.bind(window.View);
      window.View.renderResult = function (model, pred, ms, extraNote) {
        // Call original to keep UI behavior
        const ret = _orig(model, pred, ms, extraNote);
        // Then mirror into the table
        try {
          const classes = (model && model.meta && Array.isArray(model.meta.classes))
            ? model.meta.classes : [1, 2, 3];
          const fused = (pred && Array.isArray(pred.Pfused)) ? pred.Pfused : [];
          const sim   = (pred && Array.isArray(pred.S))      ? pred.S      : [];
          // If auto-rejected / uncertain with zeros, show note
          const uncertain = (pred && pred.label === 'Uncertain');
          const allZeros = (!fused.length || fused.every(v => !v)) &&
                           (!sim.length   || sim.every(v => !v));
          const hasRejectMsg = typeof extraNote === 'string' &&
                               /Auto-?rejected/i.test(extraNote);

          if (hasRejectMsg || (uncertain && allZeros)) {
            clearNote("Auto-rejected: missing rate > 30%");
          } else if (fused.length || sim.length) {
            fillTable(classes, fused, sim);
          }
        } catch (e) {
          console.error("result_filler: wrapper error:", e);
        }
        return ret;
      };
    }
  } catch (e) {
    console.warn("result_filler: could not wrap View.renderResult:", e);
  }

  // ---------- Fallback: observe text changes and parse ----------
  const NUM = String.raw`[+-]?(?:\\d+(?:\\.\\d+)?|\\.\\d+)(?:e[+-]?\\d+)?`;
  const rowRe = new RegExp(String.raw`^(\\d+)\\s+(${NUM})\\s+(${NUM})$`, "i");

  function parseTextAndFill(txt) {
    // Auto-reject?
    if (/Auto-?rejected/i.test(txt)) {
      clearNote("Auto-rejected: missing rate > 30%");
      return true;
    }
    // Find header line that starts with "Class"
    const lines = (txt || "").split(/\r?\n/).map(s => s.trim());
    const headerIdx = lines.findIndex(s => /^Class\b/i.test(s));
    if (headerIdx < 0) return false;

    const fused = [];
    const sim = [];
    for (let i = 1; i <= 3; i++) {
      const line = lines[headerIdx + i] || "";
      const m = line.match(rowRe);
      if (m) {
        fused.push(parseFloat(m[2]));
        sim.push(parseFloat(m[3]));
      }
    }
    if (fused.length) {
      fillTable([1, 2, 3], fused, sim);
      return true;
    }
    return false;
  }

  // Try once immediately
  parseTextAndFill(resultBox.textContent || "");

  // Watch for future updates of the text block
  const obs = new MutationObserver(() => {
    parseTextAndFill(resultBox.textContent || "");
  });
  obs.observe(resultBox, { childList: true, subtree: true, characterData: true });

  // Initialize with default note if empty
  if (!tbody.children.length) clearNote();
})();
