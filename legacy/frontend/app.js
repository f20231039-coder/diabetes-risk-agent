// API Configuration
const API_BASE_URL = 'http://localhost:5000';

// DOM Elements
const predictionForm = document.getElementById('predictionForm');
const predictBtn = document.getElementById('predictBtn');
const btnText = predictBtn.querySelector('.btn-text');
const spinner = predictBtn.querySelector('.spinner');
const errorMessage = document.getElementById('errorMessage');
const errorText = document.getElementById('errorText');
const resultArea = document.getElementById('resultArea');

// Form input elements
const glucoseInput = document.getElementById('glucose');
const insulinInput = document.getElementById('insulin');
const bmiInput = document.getElementById('bmi');
const ageInput = document.getElementById('age');
const modelSelect = document.getElementById('model');

/**
 * Validate user input before sending to API
 * @param {Object} data - Form data to validate
 * @returns {Object} - { valid: boolean, errors: string[] }
 */
function validateInput(data) {
    const errors = [];

    // Validate Glucose (allow 0 as valid value)
    if (data.glucose === undefined || data.glucose === null || data.glucose === '' || isNaN(data.glucose)) {
        errors.push('Glucose level is required and must be a number');
    } else if (data.glucose < 0 || data.glucose > 300) {
        errors.push('Glucose level must be between 0 and 300 mg/dL');
    }

    // Validate Insulin (allow 0 as valid value - common in dataset)
    if (data.insulin === undefined || data.insulin === null || data.insulin === '' || isNaN(data.insulin)) {
        errors.push('Insulin level is required and must be a number');
    } else if (data.insulin < 0 || data.insulin > 1000) {
        errors.push('Insulin level must be between 0 and 1000 μU/mL');
    }

    // Validate BMI (allow 0 as valid value)
    if (data.bmi === undefined || data.bmi === null || data.bmi === '' || isNaN(data.bmi)) {
        errors.push('BMI is required and must be a number');
    } else if (data.bmi < 10 || data.bmi > 70) {
        errors.push('BMI must be between 10 and 70');
    }

    // Validate Age (must be at least 1)
    if (data.age === undefined || data.age === null || data.age === '' || isNaN(data.age)) {
        errors.push('Age is required and must be a number');
    } else if (data.age < 1 || data.age > 120) {
        errors.push('Age must be between 1 and 120 years');
    } else if (!Number.isInteger(Number(data.age))) {
        errors.push('Age must be a whole number');
    }

    // Validate Model
    if (!data.model || (data.model !== 'naive_bayes' && data.model !== 'mlp')) {
        errors.push('Please select a valid prediction model');
    }

    return {
        valid: errors.length === 0,
        errors: errors
    };
}

/**
 * Send prediction request to Flask API
 * @param {Object} data - Patient data
 * @returns {Promise<Object>} - API response
 */
async function sendPredictionRequest(data) {
    const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    });

    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || errorData.message || 'Prediction failed');
    }

    return await response.json();
}

/**
 * Display prediction result to user
 * @param {Object} result - API response with prediction
 */
function displayResult(result) {
    // Hide error message if visible
    hideError();

    // Get result elements
    const resultIcon = document.getElementById('resultIcon');
    const resultTitle = document.getElementById('resultTitle');
    const resultMessage = document.getElementById('resultMessage');
    const predictionValue = document.getElementById('predictionValue');
    const probabilityValue = document.getElementById('probabilityValue');
    const modelUsed = document.getElementById('modelUsed');
    const resultCard = resultArea.querySelector('.result-card');

    // Determine if diabetes is predicted
    const hasDiabetes = result.prediction === 1;
    const probability = (result.probability * 100).toFixed(1);

    // Update result content
    if (hasDiabetes) {
        resultIcon.textContent = '⚠️';
        resultTitle.textContent = 'Diabetes Risk Detected';
        resultMessage.textContent = 'The model predicts a positive diabetes outcome. Please consult with a healthcare professional for proper diagnosis and treatment.';
        resultCard.className = 'result-card result-positive';
        predictionValue.textContent = 'Positive (1)';
    } else {
        resultIcon.textContent = '✅';
        resultTitle.textContent = 'No Diabetes Detected';
        resultMessage.textContent = 'The model predicts a negative diabetes outcome. However, regular health checkups are still recommended.';
        resultCard.className = 'result-card result-negative';
        predictionValue.textContent = 'Negative (0)';
    }

    probabilityValue.textContent = `${probability}%`;
    
    // Format model name for display
    // The API returns formatted names like "Mlp" or "Naive Bayes"
    const modelNameLower = result.model_used.toLowerCase();
    let modelName;
    if (modelNameLower.includes('mlp')) {
        modelName = 'MLP (Multilayer Perceptron)';
    } else if (modelNameLower.includes('naive')) {
        modelName = 'Naive Bayes';
    } else {
        modelName = result.model_used; // Fallback to whatever API returns
    }
    modelUsed.textContent = modelName;

    // Show result area with animation
    resultArea.style.display = 'block';
    resultArea.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Display error message to user
 * @param {string} message - Error message to display
 */
function displayError(message) {
    errorText.textContent = message;
    errorMessage.style.display = 'flex';
    resultArea.style.display = 'none';
    
    // Scroll to error message
    errorMessage.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Hide error message
 */
function hideError() {
    errorMessage.style.display = 'none';
}

/**
 * Show loading state on submit button
 */
function showLoading() {
    predictBtn.disabled = true;
    btnText.textContent = 'Processing...';
    spinner.style.display = 'inline-block';
}

/**
 * Hide loading state on submit button
 */
function hideLoading() {
    predictBtn.disabled = false;
    btnText.textContent = 'Predict';
    spinner.style.display = 'none';
}

/**
 * Handle form submission
 * @param {Event} event - Form submit event
 */
async function handleFormSubmit(event) {
    event.preventDefault();

    // Hide previous results and errors
    hideError();
    resultArea.style.display = 'none';

    // Collect form data
    const formData = {
        glucose: parseFloat(glucoseInput.value),
        insulin: parseFloat(insulinInput.value),
        bmi: parseFloat(bmiInput.value),
        age: parseInt(ageInput.value),
        model: modelSelect.value
    };

    // Validate input
    const validation = validateInput(formData);
    if (!validation.valid) {
        displayError(validation.errors.join('. '));
        return;
    }

    // Show loading state
    showLoading();

    try {
        // Send prediction request
        const result = await sendPredictionRequest(formData);
        
        // Display result
        displayResult(result);
    } catch (error) {
        // Display error
        displayError(error.message || 'Failed to connect to the prediction service. Please ensure the Flask API is running on http://localhost:5000');
    } finally {
        // Hide loading state
        hideLoading();
    }
}

/**
 * Handle form reset
 */
function handleFormReset() {
    hideError();
    resultArea.style.display = 'none';
}

// Event Listeners
predictionForm.addEventListener('submit', handleFormSubmit);
predictionForm.addEventListener('reset', handleFormReset);

// Add real-time validation feedback
[glucoseInput, insulinInput, bmiInput, ageInput].forEach(input => {
    input.addEventListener('input', () => {
        if (input.value && !input.checkValidity()) {
            input.classList.add('invalid');
        } else {
            input.classList.remove('invalid');
        }
    });
});

// Log initialization
console.log('Diabetes Prediction System initialized');
console.log(`API endpoint: ${API_BASE_URL}/predict`);
