
let model = null;
let scaler = null;
let imputer = null;
let feature_names = [];

document.addEventListener("DOMContentLoaded", async function () {
    try {
        model = await fetch("model.json").then(res => res.json());
        scaler = await fetch("scaler.json").then(res => res.json());
        imputer = await fetch("imputer.json").then(res => res.json());

        if (!scaler.mean || !scaler.scale) throw new Error("scaler 格式错误");
        feature_names = Object.keys(scaler.mean)

        const inputDiv = document.getElementById("inputs");
        feature_names.forEach(name => {
            const label = document.createElement("label");
            label.textContent = name + ": ";
            const input = document.createElement("input");
            input.name = name;
            input.placeholder = name;
            inputDiv.appendChild(label);
            inputDiv.appendChild(input);
            inputDiv.appendChild(document.createElement("br"));
        });

    } catch (err) {
        alert("❌ 模型或参数加载失败： " + err.message);
        return;
    }

    function imputeMissing(value, mean) {
        return value === "" || value === null || isNaN(value) ? mean : parseFloat(value);
    }

    function standardize(value, mean, std) {
        return std === 0 ? 0 : (value - mean) / std;
    }

   function predictSingle(inputData) {
    if (!model || !scaler || !imputer) throw new Error("模型未加载");

    const imputed = feature_names.map(name =>
        imputeMissing(inputData[name], imputer[name])
    );

    const standardized = imputed.map((val, i) =>
        standardize(val, scaler.mean[feature_names[i]], scaler.scale[feature_names[i]])
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
            inputData[input.name] = input.value.trim();
        });

        try {
            const prediction = predictSingle(inputData);
            document.getElementById("result").innerText = "Predicted Cluster: Cluster " + prediction;
        } catch (error) {
            document.getElementById("result").innerText = "❌ Prediction failed: " + error.message;
        }
    });

    document.getElementById("uploadBtn").addEventListener("click", function () {
        const file = document.getElementById("csvFile").files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = function (e) {
            const lines = e.target.result.trim().split("\n");
            const headers = lines[0].split(",");

            const output = [];

            for (let i = 1; i < lines.length; i++) {
                const row = lines[i].split(",");
                const inputData = {};
                headers.forEach((h, idx) => {
                    inputData[h.trim()] = row[idx].trim();
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

