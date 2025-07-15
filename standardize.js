const meanStd = {
  "Age": {
    "mean": -0.033301,
    "std": 0.986851
  },
  "Female": {
    "mean": 0.011314,
    "std": 0.988313
  },
  "Fever": {
    "mean": -0.01091,
    "std": 0.988899
  },
  "Weight.loss": {
    "mean": -0.014079,
    "std": 0.96037
  },
  "Proximal.pain": {
    "mean": -0.001981,
    "std": 0.996899
  },
  "Peripheral.arthritis": {
    "mean": 0.009713,
    "std": 1.00578
  },
  "Headache": {
    "mean": -0.008073,
    "std": 0.98171
  },
  "Jaw.claudication": {
    "mean": -0.001358,
    "std": 1.008839
  },
  "Visual.symptoms": {
    "mean": -0.013581,
    "std": 0.92938
  },
  "ESR": {
    "mean": -0.013202,
    "std": 1.006397
  },
  "CRP": {
    "mean": -0.030787,
    "std": 0.973137
  },
  "WBC": {
    "mean": -0.0049,
    "std": 1.016203
  },
  "Hb": {
    "mean": 0.018187,
    "std": 0.999556
  },
  "PLT": {
    "mean": -0.015323,
    "std": 0.980066
  },
  "Neutrophils": {
    "mean": -0.020592,
    "std": 1.00303
  },
  "Lymphocytes": {
    "mean": 0.0044,
    "std": 1.002098
  },
  "Hct": {
    "mean": 0.011315,
    "std": 1.00024
  },
  "RDW": {
    "mean": 0.011022,
    "std": 0.994255
  },
  "RF": {
    "mean": -0.000795,
    "std": 1.005044
  },
  "CCP": {
    "mean": 0.019808,
    "std": 1.062793
  },
  "ANA": {
    "mean": -0.023171,
    "std": 0.966524
  },
  "ALT": {
    "mean": 0.024358,
    "std": 1.016546
  },
  "AST": {
    "mean": 0.010255,
    "std": 0.982948
  },
  "GGT": {
    "mean": 0.014359,
    "std": 0.988518
  },
  "ALP": {
    "mean": 0.015581,
    "std": 1.017116
  },
  "Albumin": {
    "mean": 0.013119,
    "std": 1.021434
  }
};

function standardizeInput(rawInput) {
  return rawInput.map((val, idx) => {
    const key = featureList[idx];
    const stats = meanStd[key];
    return (val - stats.mean) / stats.std;
  });
}
