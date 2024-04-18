from flask import Flask, request, render_template, send_file
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import textwrap
import os

app = Flask(__name__)

# Define the path to the folder where uploaded images will be stored
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the path to the folder where downloaded OCR text files will be stored
DOWNLOAD_FOLDER = 'downloads'
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER

# Function to extract text values from a JSON object
def extract_text(data):
    text_values = []
    if isinstance(data, dict):
        for key, value in data.items():
            text_values.extend(extract_text(value))
    elif isinstance(data, list):
        for item in data:
            text_values.extend(extract_text(item))
    elif isinstance(data, str):
        # Replace '-' with '\n' and split the text into words
        words = data.replace('-', '\n').split()
        text_values.extend(words)
    return text_values

# Function to perform OCR on the uploaded image and return the OCR output as a text file
def perform_ocr(image_path):
    docs = DocumentFile.from_images(image_path)
    model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
    result = model(docs)
    json_output = result.export()

    # Extract text values
    text_values = extract_text(json_output)

    # Join the text values into a single string with spaces as separators
    formatted_text = ' '.join(text_values)

    # Wrap the text at the specified maximum width
    wrapped_text = textwrap.fill(formatted_text, width=80)

    return wrapped_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', message='No file part')
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', message='No selected file')
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Perform OCR on the uploaded image
        ocr_output = perform_ocr(file_path)

        # Save the OCR output to a text file
        output_file_path = os.path.join(app.config['DOWNLOAD_FOLDER'], 'ocr_output.txt')
        with open(output_file_path, 'w') as text_file:
            text_file.write(ocr_output)

        return send_file(output_file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
