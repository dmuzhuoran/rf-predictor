document.addEventListener("DOMContentLoaded", function () {
    const model = JSON.parse(localStorage.getItem("rf_model"));
    const scaler = JSON.parse(localStorage.getItem("scaler"));
    const imputer = JSON.parse(localStorage.getItem("imputer"));
    const feature_names = scaler.feature_names;

    function standardize(value, mean, std) {
        return std === 0 ? 0 : (value - mean) / std;
    }

    function imputeMissing(value, mean) {
        return value === "" || value === null || isNaN(value) ? mean : parseFloat(value);
    }

    function predictSingle(inputData) {
        let imputed = feature_names.map((name, i) =>
            imputeMissing(inputData[name], imputer.means[i])
        );

        let standardized = imputed.map((val, i) =>
            standardize(val, scaler.means[i], scaler.stds[i])
        );

        const class_votes = {};
        for (let tree of model.trees) {
            let node = tree;
            while (!node.is_leaf) {
                let val = standardized[node.feature];
                node = val <= node.threshold ? node.left : node.right;
            }
            const pred = node.prediction;
            class_votes[pred] = (class_votes[pred] || 0) + 1;
        }

        return Object.entries(class_votes).sort((a, b) => b[1] - a[1])[0][0];
    }

    const form = document.querySelector("form");
    form.addEventListener("submit", function (e) {
        e.preventDefault();

        const inputData = {};
        const inputs = form.querySelectorAll("input, select");
        inputs.forEach(input => {
            const name = input.name;
            const value = input.value.trim();
            inputData[name] = value;
        });

        try {
            const prediction = predictSingle(inputData);
            document.getElementById("result").innerText = "Predicted Cluster: Cluster " + prediction;
        } catch (error) {
            document.getElementById("result").innerText = "❌ Prediction failed: " + error.message;
        }
    });
});

