let model, scaler, imputer;
let feature_names = [];

document.addEventListener("DOMContentLoaded", async function () {
  try {
    // 正确加载 model、scaler、imputer
    model = await fetch("model300.json").then(res => res.json());
    scaler = await fetch("scaler.json").then(res => res.json());
    imputer = await fetch("imputer.json").then(res => res.json());
    feature_names = scaler.feature_names;
  } catch (err) {
    alert("❌ 模型加载失败：" + err.message);
    return;
  }

  // 以下为标准化 + 预测函数 + 表单处理
  const binary_vars = ["Female", "Fever", "Proximal.pain", "Peripheral.arthritis", "Headache", "Jaw.claudication", "Visual.symptoms", "RF", "CCP", "ANA"];

  function standardize(value, mean, std) {
    return std === 0 ? 0 : (value - mean) / std;
  }

  function imputeMissing(value, mean) {
    return value === "" || value === null || isNaN(value) ? mean : parseFloat(value);
  }

  function predictSingle(inputData) {
    const imputed = feature_names.map((name, i) =>
      imputeMissing(inputData[name], imputer.means[i])
    );

    const standardized = imputed.map((val, i) =>
      standardize(val, scaler.means[i], scaler.stds[i])
    );

    const class_votes = {};
    for (let tree of model.trees) {
      let node = tree;
      while (!node.is_leaf) {
        const val = standardized[node.feature];
        node = val <= node.threshold ? node.left : node.right;
      }
      const pred = node.prediction;
      class_votes[pred] = (class_votes[pred] || 0) + 1;
    }

    return Object.entries(class_votes).sort((a, b) => b[1] - a[1])[0][0];
  }

  document.querySelector("form").addEventListener("submit", function (e) {
    e.preventDefault();
    const inputData = {};
    let valid = true;

    const inputs = document.querySelectorAll("input");
    inputs.forEach(input => {
      const name = input.name;
      const value = input.value.trim();
      if (binary_vars.includes(name) && value !== "0" && value !== "1") {
        alert(name + " must be 0 or 1.");
        valid = false;
      }
      inputData[name] = value === "" ? null : parseFloat(value);
    });

    if (!valid) return;

    try {
      const prediction = predictSingle(inputData);
      document.getElementById("result").innerText = "Predicted Cluster: Cluster " + prediction;
    } catch (error) {
      document.getElementById("result").innerText = "❌ Prediction failed: " + error.message;
    }
  });
});

