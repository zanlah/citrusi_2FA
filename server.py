from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from faceID import createModel, identifyFace
app = Flask(__name__)

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
    image = request.files['image']
    user_id = request.form['userId']
    if image and user_id:
        filename = secure_filename(image.filename)

         #shrani sliko v mapo files/images, iz koder ga lahko referenciraš
        image.save(f'./files/images/{filename}')

        #funkcija kjer se preveri obraz
        identifyFace(f'./files/images/{filename}', user_id)


        return jsonify({'status': 'success', 'message': 'Image uploaded successfully'}), 200
    else:
        return jsonify({'status': 'error', 'message': 'Missing image or userId'}), 400

if __name__ == '__main__':
    app.run(debug=True)
