// --- Global Variables ---
let ortSession2yrs;
let ortSession5yrs; // Add 5-year session back
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
        statusAlert.innerHTML = 'Models loaded successfully. Ready to predict.'; // Restore original message
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

        displayResult(result2yrs, result5yrs); // Pass 5-year result again

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
 * Processes model output data (logits) to get a prediction.
 */
function getPrediction(outputData) {
    const probabilities = softmax(Array.from(outputData));
    const confidence = Math.max(...probabilities);
    const predictedIndex = probabilities.indexOf(confidence);
    return { predictedIndex, confidence };
}

/**
 * Displays the two prediction results in the UI.
 * (Modified to show 5-year result again)
 */
function displayResult(result2yrs, result5yrs) {
    
    const class2yrs = CLASS_NAMES[result2yrs.predictedIndex] || 'Unknown';
    const class5yrs = CLASS_NAMES[result5yrs.predictedIndex] || 'Unknown';
    
    const conf2yrs = (result2yrs.confidence * 100).toFixed(1);
    const conf5yrs = (result5yrs.confidence * 100).toFixed(1);

    // Get color classes
    const colorClass2yrs = getResultColor(result2yrs.predictedIndex);
    const colorClass5yrs = getResultColor(result5yrs.predictedIndex);

    // Restore HTML to show 5-year result
    resultText.innerHTML = `
        2-Year: <span class="${colorClass2yrs} fw-bold">${class2yrs}</span>
        <br>
        5-Year: <span class="${colorClass5yrs} fw-bold">${class5yrs}</span>
    `;
    
    // Restore text to show 5-year result
    resultProb.textContent = `Confidence: ${conf2yrs}% (2-Year) / ${conf5yrs}% (5-Year)`;

    resultCard.style.display = 'block';
}

/**
 * Returns a Bootstrap text color class based on the prediction index.
 */
function getResultColor(predictedIndex) {
    switch (predictedIndex) {
        case 0: // Survival
            return 'text-success';
        case 1: // TKA
            return 'text-warning';
        case 2: // HTO
            return 'text-danger';
        default:
            return 'text-dark';
    }
}

/**
 * Computes softmax for an array of numbers (logits).
 */
function softmax(arr) {
    const maxLogit = Math.max(...arr);
    const exps = arr.map(x => Math.exp(x - maxLogit));
    const sumExps = exps.reduce((a, b) => a + b);
    return exps.map(x => x / sumExps);
}

/**
 * Sets the loading state of the prediction button.
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