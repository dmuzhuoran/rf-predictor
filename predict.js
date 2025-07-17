async function loadModelComponents() {
  const [modelRes, imputerRes, scalerRes] = await Promise.all([
    fetch("model.json"),
    fetch("imputer.json"),
    fetch("scaler.json"),
  ]);
  return {
    model: await modelRes.json(),
    imputer: await imputerRes.json(),
    scaler: await scalerRes.json(),
  };
}

function preprocessInput(input, imputer, scaler, features) {
  return features.map((key) => {
    let val = parseFloat(input[key]);
    if (isNaN(val)) val = imputer[key];
    return (val - scaler.mean[key]) / scaler.sd[key];
  });
}

function predictSingleTree(tree, sample, featureIndexMap) {
  if (!tree.left && !tree.right && tree.prediction !== undefined) {
    const result = [0, 0, 0];
    result[tree.prediction - 1] = 1;
    return result;
  }
  const idx = featureIndexMap[tree.split_var];
  const val = sample[idx];
  if (val <= tree.split_point) {
    return predictSingleTree(tree.left_node, sample, featureIndexMap);
  } else {
    return predictSingleTree(tree.right_node, sample, featureIndexMap);
  }
}

function convertTree(treeNodes) {
  const nodes = {};
  treeNodes.forEach(node => {
    nodes[node.node] = {
      split_var: node.split_var,
      split_point: node.split_point,
      prediction: node.status === -1 ? node.prediction : undefined,
      left_node: node.status === 1 ? node.left : undefined,
      right_node: node.status === 1 ? node.right : undefined
    };
  });

  function buildTree(index) {
    const node = nodes[index];
    if (!node) return null;
    return {
      split_var: node.split_var,
      split_point: node.split_point,
      prediction: node.prediction,
      left_node: node.left_node ? buildTree(node.left_node) : null,
      right_node: node.right_node ? buildTree(node.right_node) : null
    };
  }
  return buildTree(1);
}

function aggregateVotes(trees_raw, sample, featureNames, n_classes) {
  const trees = trees_raw.map(convertTree);
  const featureIndexMap = Object.fromEntries(featureNames.map((f, i) => [f, i]));
  const votes = Array(n_classes).fill(0);
  trees.forEach((tree) => {
    const pred = predictSingleTree(tree, sample, featureIndexMap);
    pred.forEach((v, i) => votes[i] += v);
  });
  const total = votes.reduce((a, b) => a + b, 0);
  const probs = votes.map(v => v / total);
  return { votes, probs };
}

function displayResults(votes, probs) {
  const classes = ["Cluster 1", "Cluster 2", "Cluster 3"];
  const lines = classes.map((label, i) =>
    `${label}: ${votes[i]} votes (${(probs[i] * 100).toFixed(1)}%)`);
  const maxIdx = probs.indexOf(Math.max(...probs));
  document.getElementById("output").innerHTML = `
    <h3>Predicted: <b>${classes[maxIdx]}</b></h3>
    <pre>${lines.join("\n")}</pre>
  `;
}

async function predictFromForm() {
  const { model, imputer, scaler } = await loadModelComponents();
  const input = Object.fromEntries(
    model.feature_names.map(f => [f, document.getElementById(f).value])
  );
  const sample = preprocessInput(input, imputer, scaler, model.feature_names);
  const { votes, probs } = aggregateVotes(model.trees, sample, model.feature_names, model.n_classes);
  displayResults(votes, probs);
}
