import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from faceID import createModel, identifyFace
from flask_cors import CORS
from datetime import datetime
app = Flask(__name__)
CORS(app)

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'status': 'success', 'message': 'Server deluje!'}), 200


@app.route('/create-model', methods=['POST'])
def upload_video():
    user_id = request.form.get('userId')
    videos = request.files.getlist('video')
    if videos and user_id:
        for video in videos:
            filename = secure_filename(video.filename)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            directory = f'./files/{user_id}/register/'

            if not os.path.exists(directory):
                os.makedirs(directory)
            filepath = os.path.join(directory, f'{timestamp}_{filename}')
            video.save(filepath)

        createModel(filepath, user_id)
        #funkcija kjer se ustvari model
        createModel(f'./files/videos/{filename}', user_id)
        return jsonify({'status': 'success', 'message': 'Video uploaded successfully'}), 200
    else:
        return jsonify({'status': 'error', 'message': 'Missing video or userId'}), 400



@app.route('/check-face', methods=['POST'])
def upload_image():
    user_id = request.form.get('userId')
    image = request.files.get('image')
    if image and user_id:
        filename = secure_filename(image.filename)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        directory = f'./files/{user_id}/login/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        filepath = os.path.join(directory, f'{timestamp}_{filename}')
        image.save(filepath)
        # Function to check the face
        #identifyFace(save_path, user_id)  # Assuming this function processes the image and returns some result

        return jsonify({'status': 'success', 'message': 'Image uploaded and processed successfully'}), 200
    else:
        return jsonify({'status': 'error', 'message': 'Missing image or userId'}), 400

if __name__ == '__main__':
    app.run(debug=True)
