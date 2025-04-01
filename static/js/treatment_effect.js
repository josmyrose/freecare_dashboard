// Additional interactive functionality can go here
document.addEventListener('DOMContentLoaded', function () {
    // Initialize any default charts or values
    const defaultOutcome = 'Diarrhea_Had';
    const defaultModel = 'random_forest';

    // Run default analysis on page load
    fetch('/api/treatment-effect/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            outcome: defaultOutcome,
            model_type: defaultModel
        })
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateEffectSummary(data);
                updateEffectChart(data);
                return fetch('/api/treatment-effect/feature-importance', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        outcome: defaultOutcome,
                        model_type: defaultModel
                    })
                });
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateFeatureImportanceChart(data);
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
});

// Add any additional interactive elements here