function analyzeReview() {
    // Get the review text from the textarea
    var reviewText = document.getElementById('reviewTextarea').value;

    // Create a FormData object to send the review text
    var formData = new FormData();
    formData.append('review', reviewText);

    // Send a POST request to the server for analysis
    fetch('/predict', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        // Display the analysis result on the webpage
        document.getElementById('result').innerText = 'Prediction: ' + data.prediction;
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('result').innerText = 'Error occurred during analysis.';
    });
}
