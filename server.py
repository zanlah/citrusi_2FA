from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from faceID import createModel, identifyFace
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'status': 'success', 'message': 'Server deluje!'}), 200


@app.route('/create-model', methods=['POST'])
def upload_video():
    video = request.files['video']
    user_id = request.form['userId']
    if video and user_id:
        filename = secure_filename(video.filename)

        #shrani video v mapo files/videos, iz koder ga lahko referenciraš
        video.save(f'./files/videos/{filename}')

        #funkcija kjer se ustvari model
        createModel(f'./files/videos/{filename}', user_id)
        return jsonify({'status': 'success', 'message': 'Video uploaded successfully'}), 200
    else:
        return jsonify({'status': 'error', 'message': 'Missing video or userId'}), 400



@app.route('/check-face', methods=['POST'])
def upload_image():
    print("Headers:", request.headers)
    print("Content-Type:", request.content_type)
    print("Data Size:", len(request.data))  # Check if any data is coming through
    print("Files:", request.files)

    user_id = request.form.get('userId')
    image = request.files.get('image')
    if image and user_id:
        filename = secure_filename(image.filename)
        save_path = f'./files/images/{filename}'
        image.save(save_path)  # Save the streamed image

        # Function to check the face
        #identifyFace(save_path, user_id)  # Assuming this function processes the image and returns some result

        return jsonify({'status': 'success', 'message': 'Image uploaded and processed successfully'}), 200
    else:
        return jsonify({'status': 'error', 'message': 'Missing image or userId'}), 400

if __name__ == '__main__':
    app.run(debug=True)
