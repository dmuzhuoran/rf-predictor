
let model = null;
let scaler = null;
let imputer = null;
let feature_names = [];

document.addEventListener("DOMContentLoaded", async function () {
    try {
        model = await fetch("model.json").then(res => res.json());
        scaler = await fetch("scaler.json").then(res => res.json());
        imputer = await fetch("imputer.json").then(res => res.json());
        feature_names = scaler.feature_names;

        const inputDiv = document.getElementById("inputs");
        feature_names.forEach(name => {
            const label = document.createElement("label");
            label.textContent = name + ": ";
            const input = document.createElement("input");
            input.name = name;
            inputDiv.appendChild(label);
            inputDiv.appendChild(input);
            inputDiv.appendChild(document.createElement("br"));
        });
    } catch (err) {
        alert("❌ 模型或参数加载失败：" + err.message);
        return;
    }

    function standardize(value, mean, std) {
        return std === 0 ? 0 : (value - mean) / std;
    }

    function imputeMissing(value, mean) {
        return value === "" || value === null || isNaN(value) ? mean : parseFloat(value);
    }

    function predictSingle(inputData) {
        if (!model || !scaler || !imputer) throw new Error("模型未加载");

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
        const inputs = document.querySelectorAll("form input");
        inputs.forEach(input => {
            inputData[input.name] = parseFloat(input.value);
        });

        try {
            const prediction = predictSingle(inputData);
            document.getElementById("result").innerText = "Predicted Cluster: Cluster " + prediction;
        } catch (error) {
            document.getElementById("result").innerText = "❌ Prediction failed: " + error.message;
        }
    });

    document.getElementById("uploadBtn").addEventListener("click", function () {
        const fileInput = document.getElementById("csvFile");
        const file = fileInput.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = function (e) {
            const text = e.target.result;
            const rows = text.trim().split("\n");
            const headers = rows[0].split(",");

            const output = [];

            for (let i = 1; i < rows.length; i++) {
                const values = rows[i].split(",");
                const inputData = {};
                headers.forEach((h, idx) => {
                    inputData[h.trim()] = parseFloat(values[idx]);
                });

                try {
                    const pred = predictSingle(inputData);
                    output.push(`Row ${i}: Cluster ${pred}`);
                } catch (err) {
                    output.push(`Row ${i}: Error - ${err.message}`);
                }
            }

            document.getElementById("batchResult").innerHTML = output.map(r => `<div>${r}</div>`).join("");
        };
        reader.readAsText(file);
    });
});
