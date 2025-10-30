// --- Global Variables ---
let ortSession2yrs;
let ortSession5yrs;
let isPredicting = false;

// --- DOM Elements ---
const predictBtn = document.getElementById('predict-btn');
const statusAlert = document.getElementById('status-alert');
const resultCard = document.getElementById('result-card');
const resultText = document.getElementById('result-text');
const resultProb = document.getElementById('result-prob');


// Define the 10 input element IDs in the *exact order* the model expects.
const inputIds = [
    'input-age', 
    'input-bmi', 
    'input-sx-duration', 
    'input-pre-mhka', 
    'input-pre-mpta', 
    'input-pre-ldfa', 
    'input-icrs', 
    'input-effusion', 
    'input-bme', 
    'input-sifk'
];

// Map for output classes
// Index 0 = Survival, Index 1 = TKA, Index 2 = HTO
const CLASS_NAMES = ['Survival', 'TKA', 'HTO'];

// --- Initialization ---
window.onload = main;

/**
 * Main function to load both ONNX models sequentially.
 */
async function main() {
    try {
        // Use 'wasm' as a good default execution provider.
        const options = { executionProviders: ['wasm'] };
        
        // 1. Load both models sequentially (one after the other)
        ortSession2yrs = await ort.InferenceSession.create('./assets/etc/dl_fu2yrs_compact_best_model.onnx', options);
        console.log('2-Year model loaded.');
        
        ortSession5yrs = await ort.InferenceSession.create('./assets/etc/dl_fu5yrs_compact_best_model.onnx', options);
        console.log('5-Year model loaded.');

        // 2. Update UI
        console.log('2-Year and 5-Year models loaded successfully.');
        statusAlert.classList.remove('alert-info');
        statusAlert.classList.add('alert-success');
        statusAlert.innerHTML = 'Models loaded successfully. Ready to predict.';
        predictBtn.disabled = false;

    } catch (error) {
        console.error('Error during model initialization:', error);
        statusAlert.classList.remove('alert-info');
        statusAlert.classList.add('alert-danger');
        statusAlert.innerHTML = `<strong>Error:</strong> Failed to load models. ${error.message}`;
    }
}

// --- Event Listener ---
predictBtn.addEventListener('click', handlePrediction);

/**
 * Handles the prediction button click event.
 */
async function handlePrediction() {
    
    // Check if a prediction is already in progress
    if (isPredicting) {
        console.log('Prediction already in progress. Ignoring click.');
        return; 
    }

    // Set flag and disable button
    isPredicting = true; 
    setButtonLoading(true);
    resultCard.style.display = 'none'; 

    try {
        // 2. Get and validate inputs
        const inputValues = getInputs();
        if (inputValues === null) {
            isPredicting = false; // Reset the flag
            return;
        }
        
        // 3. Create ONNX Tensor
        const inputTensor = new ort.Tensor('float32', Float32Array.from(inputValues), [1, 10]);
        const feeds = { 'input': inputTensor }; 

        // 4. Run Inference on models SEQUENTIALLY
        console.log("Running 2-Year prediction...");
        const results2yrs = await ortSession2yrs.run(feeds);
        
        console.log("Running 5-Year prediction...");
        const results5yrs = await ortSession5yrs.run(feeds);
        console.log("Both predictions complete.");
        
        // 5. Post-process (Softmax) and Display
        const outputData2yrs = results2yrs.output.data; 
        const outputData5yrs = results5yrs.output.data;

        const result2yrs = getPrediction(outputData2yrs);
        const result5yrs = getPrediction(outputData5yrs);

        displayResult(result2yrs, result5yrs);

    } catch (error) {
        console.error('Error during prediction:', error);
        showTemporaryError(`Prediction failed: ${error.message}`);
    } finally {
        // Reset the flag and re-enable the button
        isPredicting = false;
        setButtonLoading(false);
    }
}

/**
 * Gets and validates all 10 input fields.
 * @returns {number[]|null} An array of float values, or null if validation fails.
 */
function getInputs() {
    const inputs = [];
    let allValid = true;

    for (const id of inputIds) {
        const element = document.getElementById(id);
        const value = element.value;

        if (value === "" || value === null) {
            element.classList.add('is-invalid');
            allValid = false;
        } else {
            element.classList.remove('is-invalid');
            inputs.push(parseFloat(value));
        }
    }

    if (!allValid) {
        showTemporaryError('Please fill in all required fields.');
        setButtonLoading(false); // Ensure button state is reset here as well
        return null;
    }
    return inputs;
}

/**
 * Processes model output data (logits) to get detailed predictions.
 * @param {Float32Array} outputData - The raw output logits from the model [logit_S, logit_T, logit_H].
 * @returns {object} An object with detailed prediction info.
 */
function getPrediction(outputData) {
    // 1. Get 3-class (Subgroup) probabilities
    const subgroupProbs = softmax(Array.from(outputData));
    const probSurvival = subgroupProbs[0];
    const probTKA = subgroupProbs[1];
    const probHTO = subgroupProbs[2];

    // 2. Get 2-class (Group) probabilities
    const groupInput_Fail = Math.max(probTKA, probHTO);
    const groupInput_Survival = probSurvival;
    const groupProbs = softmax([groupInput_Fail, groupInput_Survival]);
    const probGroupFail = groupProbs[0];
    const probGroupSurvival = groupProbs[1];

    // 3. Determine the primary prediction
    const overallMaxProb = Math.max(probSurvival, probTKA, probHTO);
    const primaryIndex = subgroupProbs.indexOf(overallMaxProb); // 0, 1, or 2
    const primaryClass = CLASS_NAMES[primaryIndex]; // 'Survival', 'TKA', or 'HTO'

    // 4. Calculate Fail subclass probabilities (TKA vs HTO)
    let failSubclassProbs = null;
    // Apply softmax *only* to the TKA and HTO logits
    const failLogits = [outputData[1], outputData[2]]; // [logit_TKA, logit_HTO]
    const failProbs = softmax(failLogits); // [prob_TKA_given_Fail, prob_HTO_given_Fail]
    
    failSubclassProbs = {
        tka: failProbs[0], // P(TKA | Fail)
        hto: failProbs[1]  // P(HTO | Fail)
    };

    // 5. Return all computed values
    return {
        primaryClass: primaryClass, // 'Survival', 'TKA', or 'HTO'
        primaryIndex: primaryIndex, // 0, 1, or 2
        probGroupSurvival: probGroupSurvival,
        probGroupFail: probGroupFail,
        failSubclassProbs: failSubclassProbs // {tka: number, hto: number}
    };
}


/**
 * Helper function to format the text for a single result (2-yr or 5-yr).
 * @param {object} result - The detailed prediction object from getPrediction().
 * @returns {object} An object containing {classString, probStringMain, probStringSub}.
 */
function formatResultStrings(result) {
    let classString = '';
    let probStringMain = '';
    let probStringSub = null; // Use null for survival
    const colorClass = getResultColor(result.primaryIndex);

    if (result.primaryClass === 'Survival') {
        // --- Class String for Survival ---
        classString = `<span class="${colorClass} fw-bold">Survival</span>`;
        
        // --- Prob String for Survival ---
        probStringMain = `${(result.probGroupSurvival * 100).toFixed(1)}%`;
        // probStringSub remains null

    } else {
        // --- Class String for Fail ---
        classString = `<span class="${colorClass} fw-bold">Fail (${result.primaryClass})</span>`;
        
        // --- Prob String Main for Fail ---
        probStringMain = `${(result.probGroupFail * 100).toFixed(1)}%`;
        
        // --- Prob String Sub for Fail ---
        const htoProb = (result.failSubclassProbs.hto * 100).toFixed(1);
        const tkaProb = (result.failSubclassProbs.tka * 100).toFixed(1);
        // Format: "82.3% HTO / 17.7% TKA"
        probStringSub = `${htoProb}% HTO / ${tkaProb}% TKA`;
    }
    
    return { classString, probStringMain, probStringSub };
}

/**
 * Displays the two prediction results in the UI.
 * @param {object} result2yrs - The 2-year prediction object from getPrediction().
 * @param {object} result5yrs - The 5-year prediction object from getPrediction().
 */
function displayResult(result2yrs, result5yrs) {
    
    const formatted2yrs = formatResultStrings(result2yrs);
    const formatted5yrs = formatResultStrings(result5yrs);

    // Update the result card text (Example: 2-Year: Survival)
    resultText.innerHTML = `
        2-Year: ${formatted2yrs.classString}
        <br>
        5-Year: ${formatted5yrs.classString}
    `;
    
    // Build the 2-Year confidence string
    let confString2yrs = `${formatted2yrs.probStringMain} (2-Year`;
    if (formatted2yrs.probStringSub) {
        // Fail case: Add ", 82.3% HTO / 17.7% TKA"
        confString2yrs += `, ${formatted2yrs.probStringSub}`;
    }
    confString2yrs += `)`; // Close parenthesis

    // Build the 5-Year confidence string
    let confString5yrs = `${formatted5yrs.probStringMain} (5-Year`;
    if (formatted5yrs.probStringSub) {
        // Fail case: Add ", 82.3% HTO / 17.7% TKA"
        confString5yrs += `, ${formatted5yrs.probStringSub}`;
    }
    confString5yrs += `)`; // Close parenthesis

    // Update the probability text
    resultProb.textContent = `Confidence: ${confString2yrs} / ${confString5yrs}`;

    resultCard.style.display = 'block';
}

/**
 * Returns a Bootstrap text color class based on the prediction index.
 * @param {number} predictedIndex - The index of the predicted class (0, 1, or 2).
 * @returns {string} A Bootstrap color class.
 */
function getResultColor(predictedIndex) {
    switch (predictedIndex) {
        case 0: // Survival
            return 'text-success';
        case 1: // TKA
            return 'text-danger';
        case 2: // HTO
            return 'text-warning';
        default:
            return 'text-dark';
    }
}

/**
 * Computes softmax for an array of numbers (logits).
 * @param {number[]} arr - An array of logits from the model output.
 * @returns {number[]} An array of probabilities.
 */
function softmax(arr) {
    const maxLogit = Math.max(...arr);
    const exps = arr.map(x => Math.exp(x - maxLogit));
    const sumExps = exps.reduce((a, b) => a + b);
    return exps.map(x => x / sumExps);
}

/**
 * Sets the loading state of the prediction button.
 * @param {boolean} isLoading - True to show spinner, false to show text.
 */
function setButtonLoading(isLoading) {
    if (isLoading) {
        predictBtn.disabled = true;
        predictBtn.innerHTML = '<span class="spinner" role="status"></span> Predicting...';
    } else {
        predictBtn.disabled = false;
        predictBtn.innerHTML = 'Predict';
    }
}

/**
 * Shows a temporary error message in the status alert box.
 * @param {string} message - The error message to display.
 */
function showTemporaryError(message) {
    statusAlert.classList.remove('alert-success');
    statusAlert.classList.add('alert-danger');
    statusAlert.innerHTML = `<strong>Error:</strong> ${message}`;
    
    // Hide the error and revert to success message after 5 seconds
    setTimeout(() => {
        statusAlert.classList.remove('alert-danger');
        statusAlert.classList.add('alert-success');
        // Restore success message to original
        statusAlert.innerHTML = 'Models loaded successfully. Ready to predict.';
    }, 5000);
}
