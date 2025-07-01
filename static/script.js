document.addEventListener('DOMContentLoaded', function () {

    const form = document.getElementById('sentiment-form');
    const reviewText = document.getElementById('review-text');
    const loader = document.getElementById('loader');
    const resultContainer = document.getElementById('result-container');
    const resultSentiment = document.getElementById('result-sentiment');
    const gaugeBar = document.getElementById('gauge-bar');
    const gaugeText = document.getElementById('gauge-text');

    form.addEventListener('submit', function (event) {
        event.preventDefault(); // Prevent default page reload

        const review = reviewText.value;
        if (review.trim() === '') {
            return; // Don't submit if textarea is empty
        }

        // Show loader and hide previous results
        loader.classList.remove('loader-hidden');
        resultContainer.classList.add('result-hidden');

        // Send data to the Flask backend
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ review: review }),
        })
        .then(response => response.json())
        .then(data => {
            // Hide loader
            loader.classList.add('loader-hidden');
            
            // Update UI with the result
            displayResult(data.sentiment, data.confidence);
        })
        .catch((error) => {
            console.error('Error:', error);
            loader.classList.add('loader-hidden');
            alert('An error occurred. Please try again.');
        });
    });

    function displayResult(sentiment, confidence) {
        // Update sentiment text and class for color coding
        resultSentiment.textContent = `Sentiment: ${sentiment}`;
        resultContainer.className = sentiment === 'Positive' ? 'positive' : 'negative';

        // Update confidence gauge
        gaugeBar.style.width = `${confidence}%`;
        gaugeText.textContent = `Confidence: ${confidence}%`;

        // Make the result container visible with a smooth transition
        resultContainer.classList.remove('result-hidden');
    }
});