// ========== Load model, imputer, scaler =============
let model, imputer, scaler, feature_names;

async function loadJSON() {
    [model, imputer, scaler] = await Promise.all([
        fetch("model.json").then(r => r.json()).then(data => { console.log('Model Loaded:', data); return data; }),
        fetch("imputer.json").then(r => r.json()).then(data => { console.log('Imputer Loaded:', data); return data; }),
        fetch("scaler.json").then(r => r.json()).then(data => { console.log('Scaler Loaded:', data); return data; }),
    ]);
    feature_names = model.feature_names;
    createFormInputs();
}

window.onload = loadJSON;

// ========== Create Form Fields Dynamically =============
function createFormInputs() {
    const container = document.getElementById("input-fields");
    feature_names.forEach(name => {
        const label = document.createElement("label");
        label.textContent = name + ": ";
        const input = document.createElement("input");
        input.type = "number";
        input.id = name;
        input.step = "any";
        if (["Female", "Fever", "Weight.loss", "Proximal.pain", "Peripheral.arthritis",
             "Headache", "Jaw.claudication", "Visual.symptoms", "RF", "CCP", "ANA"].includes(name)) {
            input.min = 0;
            input.max = 1;
        }
        container.appendChild(label);
        container.appendChild(input);
        container.appendChild(document.createElement("br"));
    });
}

// ========== Validate Single Sample Input =============
function validateInputs() {
    let values = feature_names.map(f => document.getElementById(f).value);
    if (values.includes('') || values.some(val => isNaN(val))) {
        alert("Please fill in all fields with valid numbers.");
        return false;
    }
    return true;
}

// ========== Predict for Single Sample =============
function predictSingleSample() {
    if (!validateInputs()) return;  // Ensure valid input before proceeding

    let values = feature_names.map(f => {
        let val = parseFloat(document.getElementById(f).value);
        return isNaN(val) ? imputer[feature_names.indexOf(f)] : val; // Use imputer to fill missing values
    });

    // Check for missing values tolerance (max 3 missing)
    let missingCount = values.filter(v => isNaN(v)).length;
    if (missingCount > 3) { // Max allowed missing values
        alert("Too many missing values. You can only have up to 3 missing values.");
        return;
    }

    // Standardize the input data
    let norm = values.map((v, i) => (v - scaler.mean[i]) / scaler.sd[i]);

    // Predict using the model
    let prediction = vote(model.trees, norm);
    document.getElementById("single-prediction").innerText = `Predicted: Cluster ${prediction}`;
}

// ========== Predict for CSV Upload =============
function predictFromCSV() {
    const input = document.getElementById("csvFile");
    if (!input.files.length) return alert("Please upload a CSV file");
    const reader = new FileReader();
    reader.onload = () => {
        try {
            let text = reader.result;
            let rows = text.trim().split("\n").map(r => r.split(","));
            let header = rows[0];

            // Clean up header (remove extra spaces and format)
            let cleanHeader = header.map(h => h.trim().replace(/\s+/g, '.'));
            console.log("Cleaned CSV Header:", cleanHeader);

            let missingColumns = feature_names.filter(f => !cleanHeader.includes(f));
            if (missingColumns.length > 0) {
                alert("Missing columns in CSV: " + missingColumns.join(", "));
                return;
            }

            let colIdx = feature_names.map(f => cleanHeader.indexOf(f));
            let data = rows.slice(1).map(row => colIdx.map((i, j) => {
                let val = parseFloat(row[i]);
                return isNaN(val) ? imputer[j] : val;  // Use imputer to fill missing values
            }));

            // Check for missing values tolerance (max 3 missing)
            let missingValuesCount = data.flat().filter(v => isNaN(v)).length;
            if (missingValuesCount > 3) {
                alert("Too many missing values. You can only have up to 3 missing values.");
                return;
            }

            // Standardize the data
            let scaled = data.map(row => row.map((v, i) => (v - scaler.mean[i]) / scaler.sd[i]));
            let predictions = scaled.map(norm => vote(model.trees, norm));
            let count = {};
            predictions.forEach(p => count[p] = (count[p] || 0) + 1);
            let result = Object.entries(count)
                .sort((a, b) => b[1] - a[1])
                .map(([cls, cnt]) => `Cluster ${cls}: ${cnt} votes (${(cnt / predictions.length * 100).toFixed(1)}%)`)
                .join("\n");
            let majority = Object.entries(count).sort((a, b) => b[1] - a[1])[0][0];
            document.getElementById("csv-prediction").innerText = `Predicted: Cluster ${majority}\n` + result;
        } catch (e) {
            alert("Invalid CSV format or missing headers.");
            console.error(e);
        }
    };
    reader.readAsText(input.files[0]);
}

// ========== Tree Voting Function =============
function vote(trees, sample) {
    let votes = new Array(model.n_classes).fill(0);
    for (let tree of trees) {
        let node = tree[0];
        while (node.status !== -1) {
            let feat_idx = feature_names.indexOf(node.split_var);
            node = sample[feat_idx] <= node.split_point ? tree[node.left - 1] : tree[node.right - 1];
        }
        votes[node.prediction - 1]++;
    }
    return votes.indexOf(Math.max(...votes)) + 1;
}
