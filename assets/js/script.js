let ortSession2yrs;
let ortSession5yrs;
let isPredicting = false;
let bmiChart = null; 
let bmiChartData = null; 

let predictBtn;
let statusAlert;
let resultCard;
let resultText;
let resultProb;
let toggleGraphBtn;
let graphCollapse; 
let graphCollapseElement; 
let resultsColumn;
let containerRight;

const inputIds = [
    'input-age', 'input-bmi', 'input-sx-duration', 'input-pre-mhka', 
    'input-pre-mpta', 'input-pre-ldfa', 'input-icrs', 'input-effusion', 
    'input-bme', 'input-sifk'
];
const CLASS_NAMES = ['Survival', 'TKA', 'HTO'];


document.addEventListener('DOMContentLoaded', main);


/**
 * Main function to initialize DOM elements and load models.
 */
async function main() {
    // 1. Initialize DOM elements *inside* main
    try {
        predictBtn = document.getElementById('predict-btn');
        statusAlert = document.getElementById('status-alert');
        resultCard = document.getElementById('result-card');
        resultText = document.getElementById('result-text');
        resultProb = document.getElementById('result-prob');
        containerRight = document.getElementById('container-right');
        toggleGraphBtn = document.getElementById('toggle-graph-btn');
        resultsColumn = document.getElementById('results-column');
        
        graphCollapseElement = document.getElementById('graph-collapse'); 
        graphCollapse = new bootstrap.Collapse(graphCollapseElement, { toggle: false }); 

        predictBtn.addEventListener('click', handlePrediction);

        // ADD EVENT LISTENERS FOR CHART
        graphCollapseElement.addEventListener('shown.bs.collapse', () => {
            renderBmiChart();
            console.log("Chart render triggered by 'shown' event.");
        });

        graphCollapseElement.addEventListener('hide.bs.collapse', () => {
            if (bmiChart) {
                bmiChart.destroy();
                bmiChart = null;
                console.log("Chart destroyed by 'hide' event.");
            }
        });

    } catch (e) {
        console.error("Error finding DOM elements:", e);
        alert("Fatal Error: Could not initialize page elements. Check HTML IDs.");
        return;
    }
    
    // 2. Register Chart.js Annotation Plugin
    if (window.ChartAnnotation) {
        Chart.register(window.ChartAnnotation);
    } else {
        console.error("Chart.js Annotation plugin not loaded!");
    }

    // 3. Load ONNX Models
    try {
        const options = { executionProviders: ['wasm'] };
        
        ortSession2yrs = await ort.InferenceSession.create('./assets/etc/dl_fu2yrs_compact_best_model.onnx', options);
        console.log('2-Year model loaded.');
        
        ortSession5yrs = await ort.InferenceSession.create('./assets/etc/dl_fu5yrs_compact_best_model.onnx', options);
        console.log('5-Year model loaded.');

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


/**
 * Handles the prediction button click event.
 */
async function handlePrediction() {
    
    if (isPredicting) {
        console.log('Prediction already in progress. Ignoring click.');
        return; 
    }

    isPredicting = true; 
    setButtonLoading(true);
    resultsColumn.style.display = 'block';
    containerRight.style.display = 'block';
    resultCard.style.display = 'none'; 
    toggleGraphBtn.style.display = 'none';
    
    // Explicitly destroy old chart
    toggleGraphBtn.style.display = 'none';
    graphCollapse.show();
    
    if (bmiChart) {
        bmiChart.destroy();
        bmiChart = null;
        console.log("Chart explicitly destroyed in handlePrediction.");
    }
    document.getElementById('bmi-analysis-text').innerHTML = "";
    bmiChartData = null; 

    try {
        const inputValues = getInputs();
        if (inputValues === null) {
            isPredicting = false; 
            return;
        }
        
        const inputTensor = new ort.Tensor('float32', Float32Array.from(inputValues), [1, 10]);
        const feeds = { 'input': inputTensor }; 

        console.log("Running 2-Year prediction...");
        const results2yrs = await ortSession2yrs.run(feeds);
        
        console.log("Running 5-Year prediction...");
        const results5yrs = await ortSession5yrs.run(feeds);
        console.log("Both predictions complete.");
        
        const outputData2yrs = results2yrs.output.data; 
        const outputData5yrs = results5yrs.output.data;

        const result2yrs = getPrediction(outputData2yrs); 
        const result5yrs = getPrediction(outputData5yrs);

        // Display results
        displayResult(result2yrs, result5yrs); 
        
        // Scroll to results
        if (window.innerWidth < 992) {
            setTimeout(() => {
                resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }, 500);
        } else {
            setTimeout(() => {
                resultsColumn.scrollTo({ top: 0, behavior: 'smooth' });
            }, 500);
        }

        await calculateBmiGraphData(inputValues, result5yrs); 
        toggleGraphBtn.style.display = 'block'; 

        if (graphCollapseElement.classList.contains('show')) {
            console.log("Graph is already shown, rendering manually.");
            renderBmiChart();
        }

    } catch (error) {
        console.error('Error during prediction:', error);
        showTemporaryError(`Prediction failed: ${error.message}`);
    } finally {
        isPredicting = false;
        setButtonLoading(false);
    }
}

/**
 * Renders the chart using global data.
 */
function renderBmiChart() {
    if (bmiChart || !bmiChartData) {
        if(bmiChart) console.log("Chart already rendered.");
        if(!bmiChartData) console.log("No chart data to render.");
        return;
    }

    console.log("Rendering BMI Chart...");
    const ctx = document.getElementById('bmi-chart').getContext('2d');
    
    bmiChart = new Chart(ctx, bmiChartData);
}


/**
 * Calculates data for the BMI graph.
 * Implements 2D gradient effect using 3 datasets (Line + 2 Fills).
 */
async function calculateBmiGraphData(currentInputValues, current5yrResult) {
    const numSteps = 60; 
    const currentBMI = currentInputValues[1];
    const currentProb = current5yrResult.probGroupSurvival; 
    const threshold = 0.5;

    const bmiMin = currentBMI - 10;
    const bmiMax = currentBMI + 5;
    const bmiValues = linspace(bmiMin, bmiMax, numSteps);
    const survivalProbs = [];
    const chartDataPoints = [];

    // Run model for each BMI value
    for (let i = 0; i < bmiValues.length; i++) {
        const bmi = bmiValues[i];
        const newInputs = [...currentInputValues];
        newInputs[1] = bmi; 
        
        const inputTensor = new ort.Tensor('float32', Float32Array.from(newInputs), [1, 10]);
        const results = await ortSession5yrs.run({ 'input': inputTensor });
        
        const prediction = getPrediction(results.output.data);
        const newProb = prediction.probGroupSurvival;
        
        survivalProbs.push(newProb); 
        chartDataPoints.push({ x: bmi, y: newProb }); 
    }

    // Find threshold crossing (Decreasing)
    let thresholdBMI = null;
    for (let i = 0; i < numSteps - 1; i++) {
        const p1 = survivalProbs[i];
        const p2 = survivalProbs[i+1];
        const b1 = bmiValues[i];
        const b2 = bmiValues[i+1];

        if (!isNaN(p1) && !isNaN(p2) && (p1 >= threshold && p2 < threshold)) { 
            thresholdBMI = b1 + (b2 - b1) * (threshold - p1) / (p2 - p1);
            break; 
        }
    }
    
    console.log("Threshold (To Survive) BMI found:", thresholdBMI);

    // Create split data for 2-color fill
    const dataBlue = [];
    const dataRed = [];

    if (thresholdBMI !== null) {
        // Split data at the threshold
        dataBlue.push(...chartDataPoints.filter(p => p.x <= thresholdBMI));
        dataBlue.push({ x: thresholdBMI, y: threshold });
        dataRed.push({ x: thresholdBMI, y: threshold });
        dataRed.push(...chartDataPoints.filter(p => p.x > thresholdBMI));
    } else {
        // No threshold found, fill all with one color
        if (currentProb >= threshold) {
            dataBlue.push(...chartDataPoints);
        } else {
            dataRed.push(...chartDataPoints);
        }
    }

    // Create analysis text and annotations
    const analysisTextEl = document.getElementById('bmi-analysis-text');
    let dynamicAnnotations = {}; 
    let staticAnnotations = {
        // 0.5 threshold line
        thresholdLine: {
            type: 'line',
            yMin: threshold,
            yMax: threshold,
            borderColor: 'grey',
            borderWidth: 1.5,
            borderDash: [6, 6],
            label: { content: '50% Threshold', enabled: true, position: 'start', font: { weight: 'bold' } }
        },
        // Survival threshold label
        thresholdLineLabel1: {
            type: 'label',
            xValue: bmiMin,
            yValue: 0.5,
            content: 'Survival',
            // font: { size: 14, weight: 'normal' },
            color: 'gray',
            xAdjust: 25,
            yAdjust: -9
        },
        // Fail threshold label
        thresholdLineLabel2: {
            type: 'label',
            xValue: bmiMin,
            yValue: 0.5,
            content: 'Fail',
            // font: { size: 14, weight: 'normal' },
            color: 'gray',
            xAdjust: 14,
            yAdjust: 11
        },
        // 'Current' BMI vertical line
        currentBmiLine: {
            type: 'line',
            xMin: currentBMI,
            xMax: currentBMI,
            borderColor: 'grey', 
            borderWidth: 1.5,
            borderDash: [6, 6]
        },
        // 'Current' BMI point (filled circle, color based on status)
        currentBmiPoint: {
            type: 'point',
            xValue: currentBMI,
            yValue: currentProb,
            radius: 5, 
            pointStyle: 'circle', 
            backgroundColor: 'white',
            borderColor: (currentProb < threshold) ? 'rgb(255, 50, 50)' : 'rgb(0, 100, 255)',
            borderWidth: 2
        },
        // 'Current' text label
        currentBmiLabel: {
            type: 'label',
            xValue: currentBMI,
            yValue: currentProb,
            content: 'Current',
            // font: { size: 14, weight: 'normal' },
            color: 'black',
            xAdjust: (currentProb < threshold) ? 26 : -26,
            yAdjust: (currentProb < threshold) ? -11 : 11
        },
        // 'Current' value label
        currentBmiValueLabel: {
            type: 'label',
            xValue: currentBMI,
            yValue: 0,
            content: `${currentBMI.toFixed(2)}`,
            // font: { size: 14, weight: 'normal' },
            color: 'black',
            xAdjust: (currentProb < threshold) ? 20 : -20,
            yAdjust: -8
        }
    };
    if (thresholdBMI !== null) {
        // 'To Survive' vertical line
        dynamicAnnotations.thresholdBmiLine = {
            type: 'line',
            xMin: thresholdBMI,
            xMax: thresholdBMI,
            borderColor: 'grey', 
            borderWidth: 1.5,
            borderDash: [6, 6]
        };
        // 'To Survive' point
        dynamicAnnotations.thresholdBmiPoint = {
            type: 'point',
            xValue: thresholdBMI,
            yValue: threshold,
            radius: 6,
            pointStyle: 'rectRot',
            backgroundColor: 'white',
            borderColor: 'rgb(128, 0, 128)',
            borderWidth: 2
        };
        // 'To survive' text label
        dynamicAnnotations.thresholdBmiLabel = {
            type: 'label',
            xValue: thresholdBMI,
            yValue: threshold,
            content: 'To survive',
            // font: { size: 14, weight: 'normal' },
            color: 'black',
            xAdjust: (currentProb < threshold) ? -35 : 35,
            yAdjust: (currentProb < threshold) ? 12 : -12
        };
        // 'To survive' value label
        dynamicAnnotations.thresholdBmiValueLabel = {
            type: 'label',
            xValue: thresholdBMI,
            yValue: 0,
            content: `${thresholdBMI.toFixed(2)}`,
            // font: { size: 14, weight: 'normal' },
            color: 'black',
            xAdjust: (currentProb < threshold) ? -20 : 20,
            yAdjust: -8
        };
    }

    if (currentProb < threshold && thresholdBMI !== null) {
        const bmiDiff = thresholdBMI - currentBMI;
        analysisTextEl.innerHTML = `Current 5-year probability is <strong>${(currentProb*100).toFixed(1)}% (Fail)</strong>. 
            To reach the <strong>50% (Survival)</strong> threshold, a BMI change of <strong>${bmiDiff.toFixed(2)}</strong> is suggested.`;
        
        
        dynamicAnnotations.gradientArrow = {
            type: 'line',
            xMin: thresholdBMI,
            xMax: currentBMI,
            yMin: 0.9,
            yMax: 0.9,
            borderWidth: 6,
            drawTime: 'afterDatasetsDraw',
            arrowHeads: {
                start: {
                    enabled: true,
                    display: true,
                    length: 8
                }
            },
            borderColor: (context) => {
                const chart = context.chart;
                const { ctx, chartArea } = chart;
                if (!chartArea) { return 'rgba(139, 69, 19, 1)'; } // Fallback

                const xCurrentPixel = chart.scales.x.getPixelForValue(currentBMI);
                const xThresholdPixel = chart.scales.x.getPixelForValue(thresholdBMI);
                const yPixel = chart.scales.y.getPixelForValue(0.9);

                if (!chartArea || !xCurrentPixel || !xThresholdPixel || !yPixel) { 
                    return 'rgba(139, 69, 19, 1)'; // Fallback
                }
                
                width = xCurrentPixel - xThresholdPixel;
                const gradient = ctx.createLinearGradient(0, yPixel, width, yPixel);
                gradient.addColorStop(0, 'rgba(139, 69, 19, 1)');
                gradient.addColorStop(1, 'rgba(139, 69, 19, 0.5)');
                
                return gradient;
            },
            label: {
                content: `BMI: ${bmiDiff.toFixed(2)}`,
                enabled: true,
                position: 'center',
                font: { size: 12, weight: 'bold' },
                color: 'black',
                yAdjust: -15 
            },
        };
        dynamicAnnotations.gradientArrowLabel = {
            type: 'label',
            xValue: currentBMI,
            yValue: 0.9,
            content: `BMI: ${bmiDiff.toFixed(2)}`,
            // font: { size: 12, weight: 'bold' },
            color: 'black',
            xAdjust: 32,
            yAdjust: 1
        };
        // --- [END FIX] ---
    } else if (currentProb >= threshold) {
        analysisTextEl.innerHTML = `Current 5-year probability is <strong>${(currentProb*100).toFixed(1)}% (Survival)</strong>. Already in the target range.`;
    } else {
        analysisTextEl.innerHTML = "Cannot reach 50% survival probability within this BMI range.";
    }

    bmiChartData = {
        type: 'line',
        data: {
            datasets: [
                // Blue fill with vertical alpha fade
                {
                    label: 'Survival Fill',
                    data: dataBlue,
                    borderColor: 'transparent',
                    pointRadius: 0,
                    fill: 'origin',
                    backgroundColor: (context) => {
                        const chart = context.chart;
                        const {ctx, chartArea} = chart;
                        if (!chartArea) { return null; }
                        // Vertical gradient: 0.2 alpha at top, 0 alpha at bottom
                        const gradient = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
                        gradient.addColorStop(0, 'rgba(0, 100, 255, 0.25)');
                        gradient.addColorStop(1, 'rgba(0, 100, 255, 0)');
                        return gradient;
                    }
                },
                // Red fill with vertical alpha fade
                {
                    label: 'Fail Fill',
                    data: dataRed,
                    borderColor: 'transparent',
                    pointRadius: 0,
                    fill: 'origin',
                    backgroundColor: (context) => {
                        const chart = context.chart;
                        const {ctx, chartArea} = chart;
                        if (!chartArea) { return null; }
                        // Vertical gradient: 0.2 alpha at top, 0 alpha at bottom
                        const gradient = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
                        gradient.addColorStop(0, 'rgba(255, 0, 0, 0.25)');
                        gradient.addColorStop(1, 'rgba(255, 0, 0, 0)');
                        return gradient;
                    }
                },
                // Purple line
                {
                    label: '5-Year Survival Probability',
                    data: chartDataPoints, 
                    // borderColor: 'rgb(128, 0, 128)',
                    borderColor: (context) => {
                        const chart = context.chart;
                        const {ctx, chartArea} = chart;
                        if (!chartArea) { return null; }
                        // Vertical gradient for line color
                        const gradient = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
                        gradient.addColorStop(0.3, 'rgb(0, 100, 255)');
                        gradient.addColorStop(0.5, 'rgb(128, 0, 128)');
                        gradient.addColorStop(0.7, 'rgb(255, 50, 50)');
                        return gradient;
                    },
                    borderWidth: 2,
                    tension: 0.1, 
                    pointRadius: 0, 
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: { min: 0, max: 1.0, title: { display: true, text: 'Survival Probability' } },
                x: { 
                    type: 'linear', 
                    title: { display: true, text: 'BMI' },
                    min: bmiMin,
                    max: bmiMax
                }
            },
            plugins: {
                legend: { display: false },
                title: { display: true, text: '5-Year Survival Probability vs. BMI' },
                tooltip: {
                    enabled: true,
                    mode: 'index',
                    intersect: false,
                    position: 'nearest',
                    filter: function(tooltipItem) {
                        return tooltipItem.datasetIndex === 2; // '5-Year Survival Probability'
                    },
                    callbacks: {
                        title: function(tooltipItems) {
                            if (tooltipItems[0]) {
                                return `BMI: ${parseFloat(tooltipItems[0].parsed.x).toFixed(2)}`;
                            }
                            return '';
                        },
                        label: function(tooltipItem) {
                            return `Prob: ${parseFloat(tooltipItem.parsed.y).toFixed(3)}`;
                        }
                    }
                },
                annotation: {
                    drawTime: 'afterDatasetsDraw',
                    annotations: {
                        ...staticAnnotations,
                        ...dynamicAnnotations
                    }
                }
            }
        }
    };
}


/**
 * Helper function: Creates an array of numbers.
 */
function linspace(start, stop, num) {
    const step = (stop - start) / (num - 1);
    return Array.from({ length: num }, (_, i) => start + step * i);
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
        setButtonLoading(false); 
        return null;
    }
    return inputs;
}


/**
 * Processes model output data for *both* the main text display and the graph.
 */
function getPrediction(outputData) {
    // 1. Get 3-class (Subgroup) outputs (before softmax)
    const subgroupProbs = Array.from(outputData);
    const probSurvival = subgroupProbs[0];
    const probTKA = subgroupProbs[1];
    const probHTO = subgroupProbs[2];

    // 2. Get 2-class (Group) probabilities (THIS IS THE UNIFIED LOGIC)
    const groupInput_Fail = Math.max(probTKA, probHTO);
    const groupInput_Survival = probSurvival;
    const groupProbs = softmax([groupInput_Fail, groupInput_Survival]);
    const probGroupFail = groupProbs[0];
    const probGroupSurvival = groupProbs[1];

    // 3. Determine the primary prediction
    const overallMaxProb = Math.max(probSurvival, probTKA, probHTO);
    const primaryIndex = subgroupProbs.indexOf(overallMaxProb); 
    const primaryClass = CLASS_NAMES[primaryIndex]; 

    // 4. Calculate Fail subclass probabilities (TKA vs HTO)
    let failSubclassProbs = null;
    const failLogits = [outputData[1], outputData[2]]; 
    const failProbs = softmax(failLogits); 
    
    failSubclassProbs = {
        tka: failProbs[0], // P(TKA | Fail)
        hto: failProbs[1]  // P(HTO | Fail)
    };

    // 5. Return all computed values
    return {
        primaryClass: primaryClass,
        primaryIndex: primaryIndex,
        probGroupSurvival: probGroupSurvival,
        probGroupFail: probGroupFail,
        failSubclassProbs: failSubclassProbs
    };
}


/**
 * Helper function to format the text for a single result (2-yr or 5-yr).
 */
function formatResultStrings(result) {
    let classString = '';
    let probStringMain = '';
    let probStringSub = null; 
    const colorClass = getResultColor(result.primaryIndex);

    if (result.primaryClass === 'Survival') {
        classString = `<span class="${colorClass} fw-bold">Survival</span>`;
        probStringMain = `${(result.probGroupSurvival * 100).toFixed(1)}%`;
    } else {
        classString = `<span class="${colorClass} fw-bold">Fail (${result.primaryClass})</span>`;
        probStringMain = `${(result.probGroupFail * 100).toFixed(1)}%`;
        const htoProb = (result.failSubclassProbs.hto * 100).toFixed(1);
        const tkaProb = (result.failSubclassProbs.tka * 100).toFixed(1);
        probStringSub = `${htoProb}% HTO / ${tkaProb}% TKA`;
    }
    
    return { classString, probStringMain, probStringSub };
}


/**
 * Displays the two prediction results in the UI.
 */
function displayResult(result2yrs, result5yrs) {
    
    const formatted2yrs = formatResultStrings(result2yrs);
    const formatted5yrs = formatResultStrings(result5yrs);

    resultText.innerHTML = `
        2-Year: ${formatted2yrs.classString}
        <br>
        5-Year: ${formatted5yrs.classString}
    `;
    
    let confString2yrs = `${formatted2yrs.probStringMain} (2-Year`;
    if (formatted2yrs.probStringSub) {
        confString2yrs += `, ${formatted2yrs.probStringSub}`;
    }
    confString2yrs += `)`;

    let confString5yrs = `${formatted5yrs.probStringMain} (5-Year`;
    if (formatted5yrs.probStringSub) {
        confString5yrs += `, ${formatted5yrs.probStringSub}`;
    }
    confString5yrs += `)`;

    // Use non-breaking spaces and non-breaking hyphens for better display
    const finalConf2yrs = confString2yrs.replace(/ /g, '\u00A0').replace(/-/g, '\u2011');
    const finalConf5yrs = confString5yrs.replace(/ /g, '\u00A0').replace(/-/g, '\u2011');
    resultProb.textContent = `Confidence: ${finalConf2yrs} / ${finalConf5yrs}`;

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
            return 'text-danger';
        case 2: // HTO
            return 'text-warning';
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
        statusAlert.innerHTML = 'Models loaded successfully. Ready to predict.';
    }, 5000);
}
