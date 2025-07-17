
async function loadJSON(path) {
  const response = await fetch(path);
  return await response.json();
}

function standardizeRow(row, scaler, imputer, featureNames) {
  return featureNames.map(name => {
    let val = Number(row[name]);
    if (isNaN(val)) val = imputer[name];
    return (val - scaler.mean[name]) / scaler.sd[name];
  });
}

function traverseTree(tree, sample, features) {
  let i = 0;
  while (tree[i].status !== -1) {
    const node = tree[i];
    const fIdx = features.indexOf(node.split_var);
    const splitVal = Number(node.split_point);
    const value = Number(sample[fIdx]);
    if (value <= splitVal) {
      i = Number(node.left) - 1;
    } else {
      i = Number(node.right) - 1;
    }
  }
  return Number(tree[i].prediction);
}

async function predictCSV() {
  const fileInput = document.getElementById("csvFile");
  if (!fileInput.files.length) return alert("Please select a CSV file.");

  const [model, scaler, imputer] = await Promise.all([
    loadJSON("model.json"),
    loadJSON("scaler.json"),
    loadJSON("imputer.json")
  ]);

  const file = fileInput.files[0];
  const text = await file.text();
  const lines = text.trim().split("\n");
  const headers = lines[0].split(",").map(h => h.trim());

  const results = [];
  for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(",");
    const row = Object.fromEntries(headers.map((h, j) => [h, values[j]]));

    const sample = standardizeRow(row, scaler, imputer, model.feature_names);
    const votes = Array(model.n_classes).fill(0);
    for (let tree of model.trees) {
      const pred = traverseTree(tree, sample, model.feature_names);
      if (pred >= 1 && pred <= model.n_classes) {
        votes[pred - 1]++;
      }
    }

    const predicted = votes.indexOf(Math.max(...votes)) + 1;
    results.push({ index: i, prediction: predicted });
  }

  const output = document.getElementById("output");
  let html = "<table><tr><th>Sample</th><th>Predicted Cluster</th></tr>";
  results.forEach(r => {
    html += `<tr><td>${r.index}</td><td>${r.prediction}</td></tr>`;
  });
  html += "</table>";
  output.innerHTML = html;
}
