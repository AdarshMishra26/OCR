# Flask OCR Application

This Flask application allows users to upload images containing text, perform OCR (Optical Character Recognition) on the uploaded images, and download the extracted text as a text file.

## Prerequisites

- Python
- Flask
- Doctr

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your_username/flask-ocr.git
    ```

2. Navigate to the project directory:

    ```bash
    cd flask-ocr
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Flask application:

    ```bash
    python app.py
    ```

2. Open your web browser and go to `http://localhost:5000`.

3. Upload an image file containing text.

4. Click on the "Upload" button to perform OCR on the uploaded image.

5. Download the extracted text as a text file.

## File Structure

- `app.py`: Contains the Flask application code.
- `templates/index.html`: HTML template for the upload page.
- `uploads/`: Folder to store uploaded image files.
- `downloads/`: Folder to store downloaded OCR text files.

## How it Works

- The application uses Flask, a Python web framework, to create a web server.
- Users can upload images containing text through a web interface.
- When an image is uploaded, the application performs OCR (Optical Character Recognition) using the Doctr library.
- The extracted text is then saved to a text file, which users can download.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
