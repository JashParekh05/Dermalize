import os
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Define the path to the "uploads" folder in your project directory
uploads_folder = os.path.join(os.path.dirname(__file__), 'uploads')

# Ensure the "uploads" folder exists, create it if not
if not os.path.exists(uploads_folder):
    os.makedirs(uploads_folder)

# Define a route to display the image upload form
@app.route('/')
def upload_form():
    return render_template('upload.html')

# Define a route to handle image uploads
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Save the uploaded image to the "uploads" folder
        filename = file.filename
        file_path = os.path.join(uploads_folder, filename)
        file.save(file_path)

        return jsonify({'message': 'Image uploaded successfully'})
        print("hello is this working")
        print("hello is this working")
        print("hello is this working")
        print("hello is this working")

    return jsonify({'error': 'Unexpected error'})

if __name__ == '__main__':
    app.run()
