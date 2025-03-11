const fileInput = document.getElementById('fileInput');
const dropArea = document.getElementById('dropArea');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const uploadButton = document.getElementById('uploadButton');
const resultContainer = document.getElementById('resultContainer');
const wasteCategory = document.getElementById('wasteCategory');
const subCategory = document.getElementById('subCategory');
const probability = document.getElementById('probability');
const predictedClass = document.getElementById('predictedClass');
const resultImage = document.getElementById('resultImage');

// Drag and Drop Handling
dropArea.addEventListener('click', () => fileInput.click());
dropArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropArea.classList.add('bg-white/20');
});
dropArea.addEventListener('dragleave', () => dropArea.classList.remove('bg-white/20'));
dropArea.addEventListener('drop', (e) => {
    e.preventDefault();
    dropArea.classList.remove('bg-white/20');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        fileInput.files = files; // This won't trigger 'change' event, so manually process
        showPreview(files[0]);
    }
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) showPreview(fileInput.files[0]);
});

function showPreview(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file.');
        return;
    }

    const reader = new FileReader();
    reader.onload = () => {
        previewImage.src = reader.result;
        previewContainer.classList.remove('hidden');
        previewImage.classList.add('opacity-100');
        uploadButton.disabled = false;
    };
    reader.readAsDataURL(file);
}

uploadButton.addEventListener('click', async () => {
    if (!fileInput.files.length) {
        alert('No file selected.');
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    uploadButton.textContent = "Detecting...";
    uploadButton.disabled = true;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const data = await response.json();
            const prob = parseFloat(data.probability);

            if (prob > 0.5) {
                // wasteCategory.textContent = data.category || 'Unknown';
                // subCategory.textContent = data.subcategory || 'Unknown';
                // probability.textContent = (prob * 100).toFixed(2) ;
                // predictedClass.textContent = data.predicted_class || 'N/A';
                
                // Uncomment this if an image URL is provided in the response
                // resultImage.src = data.image_url;

                resultContainer.innerHTML = `<h3 class="text-white text-xl font-semibold"><span id="wasteCategory">${data.category || 'Unknown'}</span></h3>
                <p class="text-white mt-2 opacity-90 font-semibold">Waste Type: <span id="subCategory">${data.subcategory || 'Unknown'}</span></p>
                <p class="text-white mt-2 opacity-90 font-semibold">Label: <span id="predictedClass">${data.predicted_class || 'N/A'}</span></p>
                <p class="text-white mt-2 opacity-90">Confidence: <span id="probability">${(prob * 100).toFixed(2)}</span>%</p>`

                resultContainer.classList.remove('hidden');
            } else {
                resultContainer.classList.remove('hidden');
                resultContainer.innerHTML = `<h3 class="text-white text-xl font-semibold">Can't Detect the Waste Category</h3>`;
            }
        } else {
            alert('Failed to get a response from the server.');
        }
    } catch (error) {
        alert('Error detecting waste category.');
        console.error(error);
    }

    uploadButton.textContent = "Upload & Detect";
    uploadButton.disabled = false;
});
