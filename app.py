from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
from speech_to_text import transcribe_audio_folder
import threading
import logging

app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'mp3'}

# Configure app settings
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Ensure directories exist
for directory in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
    if not os.path.exists(directory):
        os.makedirs(directory)
    logger.info(f"Directory {directory} is ready")

# Global variables for tracking transcription status
current_status = {
    'is_processing': False,
    'progress': 0,
    'message': '',
    'error': None
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        if 'files[]' not in request.files:
            logger.error("No files part in the request")
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files[]')
        if not files or all(not file.filename for file in files):
            logger.error("No selected files")
            return jsonify({'error': 'No files selected'}), 400
        
        # Clear upload directory
        try:
            for f in os.listdir(app.config['UPLOAD_FOLDER']):
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], f))
        except Exception as e:
            logger.error(f"Error clearing upload directory: {str(e)}")
        
        # Save new files
        saved_files = []
        for file in files:
            if file and file.filename and allowed_file(file.filename):
                try:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    
                    # Verify file was saved
                    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                        saved_files.append(filename)
                        logger.info(f"Successfully saved file: {filename}")
                    else:
                        logger.error(f"File {filename} was not saved properly")
                except Exception as e:
                    logger.error(f"Error saving file {file.filename}: {str(e)}")
            else:
                logger.warning(f"Invalid file: {file.filename}")
        
        if not saved_files:
            return jsonify({'error': 'No valid files were uploaded'}), 400
        
        return jsonify({
            'message': f'Successfully uploaded {len(saved_files)} files',
            'files': saved_files
        })
    
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': f'Server error during upload: {str(e)}'}), 500

@app.route('/transcribe', methods=['POST'])
def start_transcription():
    if current_status['is_processing']:
        return jsonify({'error': 'Transcription already in progress'}), 400
    
    data = request.json
    model_size = data.get('model_size', 'small')
    custom_prompt = data.get('custom_prompt', '')
    
    output_file = os.path.join(OUTPUT_FOLDER, 'transcriptions.tsv')
    
    def transcribe_task():
        try:
            current_status.update({
                'is_processing': True,
                'progress': 0,
                'message': 'Starting transcription...',
                'error': None
            })
            
            df = transcribe_audio_folder(
                UPLOAD_FOLDER,
                output_file,
                model_size=model_size,
                custom_prompt=custom_prompt
            )
            
            current_status.update({
                'is_processing': False,
                'progress': 100,
                'message': 'Transcription completed successfully!',
                'error': None
            })
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            current_status.update({
                'is_processing': False,
                'progress': 0,
                'message': 'Error during transcription',
                'error': str(e)
            })
    
    thread = threading.Thread(target=transcribe_task)
    thread.start()
    
    return jsonify({'message': 'Transcription started'})

@app.route('/status')
def get_status():
    return jsonify(current_status)

@app.route('/download')
def download_results():
    output_file = os.path.join(OUTPUT_FOLDER, 'transcriptions.tsv')
    if os.path.exists(output_file):
        return send_file(output_file, as_attachment=True)
    return jsonify({'error': 'No transcription file available'}), 404

if __name__ == '__main__':
    app.run(debug=True)