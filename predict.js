
let model, scaler, imputer;

fetch("model.json").then(res => res.json()).then(data => model = data);
fetch("scaler.json").then(res => res.json()).then(data => scaler = data);
fetch("imputer.json").then(res => res.json()).then(data => imputer = data);

function handleCSV() {
  const input = document.getElementById("csvFileInput");
  if (!input.files.length) return alert("Please upload a CSV file");

  const reader = new FileReader();
  reader.onload = () => {
    const rows = reader.result.split("\n").filter(line => line.trim()).map(line => line.split(","));
    const headers = rows[0];
    const data = rows.slice(1).map(r => Object.fromEntries(headers.map((h, i) => [h.trim(), parseFloat(r[i])])));

    const standardized = data.map(row => {
      const obj = {};
      for (let key of model.feature_names) {
        let val = row[key];
        if (isNaN(val)) val = imputer[key];
        obj[key] = (val - scaler.mean[key]) / scaler.sd[key];
      }
      return obj;
    });

    const predictions = standardized.map(row => {
      const votes = model.trees.map(tree => traverseTree(tree, row));
      return mostCommon(votes);
    });

    const table = `
      <table><tr><th>Sample</th><th>Predicted Cluster</th></tr>
      ${predictions.map((c, i) => `<tr><td>${i + 1}</td><td>${c}</td></tr>`).join("")}
      </table>`;
    document.getElementById("output").innerHTML = table;
  };
  reader.readAsText(input.files[0]);
}

function traverseTree(tree, row) {
  let node = tree[0];
  while (node.status !== -1) {
    const split = node.split_var;
    const value = node.split_point;
    node = row[split] <= value ? tree[node.left - 1] : tree[node.right - 1];
  }
  return parseInt(node.prediction);
}

function mostCommon(arr) {
  const counts = {};
  for (let v of arr) counts[v] = (counts[v] || 0) + 1;
  return Object.entries(counts).reduce((a, b) => (a[1] >= b[1] ? a : b))[0];
}
