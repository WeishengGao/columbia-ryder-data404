document.getElementById('predict-btn').addEventListener('click', function() {
    var methodSelect = document.getElementById('method-select');
    var method = methodSelect.value;

    console.log("Method:", method);

    // Send request to the server
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ method: method })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.image_url_1 && data.image_url_2 && data.image_url_3 && data.image_url_4 && data.image_url_nn) {
            // Clear previous results
            document.getElementById('result-container').innerHTML = '';

            // Display each image
            var img1 = new Image();
            img1.src = data.image_url_1;
            img1.alt = "Prediction Result - Financial Sources";

            var img2 = new Image();
            img2.src = data.image_url_2;
            img2.alt = "Prediction Result - Aggregate Totals";

            var img3 = new Image();
            img3.src = data.image_url_3;
            img3.alt = "ARIMA ACF/PACF Plot";

            var img4 = new Image();
            img4.src = data.image_url_4;
            img4.alt = "ARIMA Aggregate Totals";

            var imgNN = new Image();
            imgNN.src = data.image_url_nn;
            imgNN.alt = "Neural Network Aggregate Totals";

            // Append new images to the result container
            document.getElementById('result-container').appendChild(img1);
            document.getElementById('result-container').appendChild(img2);
            document.getElementById('result-container').appendChild(img3);
            document.getElementById('result-container').appendChild(img4);
            document.getElementById('result-container').appendChild(imgNN);
        } else {
            console.error('Error: No image URLs returned.');
        }
    })
    .catch(error => console.error('Error in fetch request:', error));
});