import cv2
import tempfile
import os
import uuid
from flask import Flask, jsonify, render_template, send_file, redirect, request
from werkzeug.utils import secure_filename
from OBR import SegmentationEngine, BrailleClassifier, BrailleImage
from googletrans import Translator
from gtts import gTTS

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
tempdir = tempfile.TemporaryDirectory()

app = Flask("Optical Braille Recognition Demo")
app.config['UPLOAD_FOLDER'] = tempdir.name

translator = Translator()

# ---- Allowed languages ----
LANGUAGES = {
    'af': 'Afrikaans',
    'ar': 'Arabic',
    'bn': 'Bengali',
    'bg': 'Bulgarian',
    'ca': 'Catalan',
    'zh-cn': 'Chinese (Simplified)',
    'zh-tw': 'Chinese (Traditional)',
    'hr': 'Croatian',
    'cs': 'Czech',
    'da': 'Danish',
    'nl': 'Dutch',
    'en': 'English',
    'et': 'Estonian',
    'fi': 'Finnish',
    'fr': 'French',
    'de': 'German',
    'el': 'Greek',
    'gu': 'Gujarati',
    'he': 'Hebrew',
    'hi': 'Hindi',
    'hu': 'Hungarian',
    'id': 'Indonesian',
    'it': 'Italian',
    'ja': 'Japanese',
    'kn': 'Kannada',
    'ko': 'Korean',
    'ml': 'Malayalam',
    'mr': 'Marathi',
    'ne': 'Nepali',
    'no': 'Norwegian',
    'fa': 'Persian',
    'pl': 'Polish',
    'pt': 'Portuguese',
    'pa': 'Punjabi',
    'ro': 'Romanian',
    'ru': 'Russian',
    'si': 'Sinhala',
    'es': 'Spanish',
    'sw': 'Swahili',
    'sv': 'Swedish',
    'ta': 'Tamil',
    'te': 'Telugu',
    'th': 'Thai',
    'tr': 'Turkish',
    'uk': 'Ukrainian',
    'ur': 'Urdu',
    'vi': 'Vietnamese',
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template("index.html", languages=LANGUAGES)

@app.route('/favicon.ico')
def fav():
    return send_file('favicon.ico', mimetype='image/ico')

@app.route('/coverimage')
def cover_image():
    return send_file('samples/sample1.png', mimetype='image/png')

@app.route('/procimage/<string:img_id>')
def proc_image(img_id):
    image = f'{tempdir.name}/{secure_filename(img_id)}-proc.png'
    if os.path.exists(image):
        return send_file(image, mimetype='image/png')
    return redirect('/coverimage')

@app.route('/digest', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": True, "message": "No file part"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": True, "message": "No selected file"})

    if file and allowed_file(file.filename):
        filename = ''.join(str(uuid.uuid4()).split('-'))
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)

        classifier = BrailleClassifier()
        img = BrailleImage(path)
        for letter in SegmentationEngine(image=img):
            letter.mark()
            classifier.push(letter)
        cv2.imwrite(f'{tempdir.name}/{filename}-proc.png', img.get_final_image())
        os.unlink(path)

        clean_text = classifier.digest().strip()

        return jsonify({
            "error": False,
            "message": "Processed successfully",
            "img_id": filename,
            "digest": clean_text
        })

@app.route('/speech', methods=['POST'])
def text_to_speech():
    data = request.get_json()
    text = data.get("text", "")
    lang_code = data.get("language", "en")

    if not text:
        return jsonify({"error": True, "message": "No text provided"})

    # Translate text
    translated = translator.translate(text, dest=lang_code).text

    # Generate speech file
    audio_filename = f"{uuid.uuid4()}.mp3"
    audio_path = os.path.join(tempdir.name, audio_filename)
    tts = gTTS(translated, lang=lang_code)
    tts.save(audio_path)

    return jsonify({
        "error": False,
        "translated": translated,
        "audio_url": f"/audio/{audio_filename}"
    })

@app.route('/audio/<string:filename>')
def serve_audio(filename):
    audio_path = os.path.join(tempdir.name, secure_filename(filename))
    if os.path.exists(audio_path):
        return send_file(audio_path, mimetype='audio/mpeg')
    return jsonify({"error": True, "message": "Audio not found"})

if __name__ == "__main__":
    app.run(debug=True)