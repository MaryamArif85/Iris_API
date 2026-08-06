document.getElementById('iris-form').addEventListener('submit', async (e) => {
    e.preventDefault(); // stop page reload

    const resultDiv = document.getElementById('result');
    resultDiv.style.display = 'none';

    const data = {
        sepal_length: parseFloat(document.getElementById('sepal_length').value),
        sepal_width: parseFloat(document.getElementById('sepal_width').value),
        petal_length: parseFloat(document.getElementById('petal_length').value),
        petal_width: parseFloat(document.getElementById('petal_width').value),
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const result = await response.json();

        resultDiv.innerHTML = `
            <h3>🌸 Prediction: ${result.predicted_label}</h3>
            <p><b>Confidence:</b> ${result.confidence}%</p>
            <p><b>Model Version:</b> ${result.model_version}</p>
        `;
        resultDiv.style.display = 'block';

    } catch (error) {
        resultDiv.innerHTML = `<p style="color:red;"><b>Error:</b> ${error.message}</p>`;
        resultDiv.style.display = 'block';
    }
});