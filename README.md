# citrusi 2FA server

Flask server that has two routes, one for creating the model, the other for identifying face.

## Setup 

     pip install Flask Werkzeug
## Run

    python server.py     

## Routes
### `/create-model` [POST]

This endpoint is used to upload a video file and associate it with a user ID to create a user authentication model.

#### Arguments:

-   `video`: The video file of the user to be uploaded.
-   `userId`: A unique identifier for the user.

#### Responses:

-   `200 OK`: Returns success message if the video is successfully uploaded and the model is created.
-   `400 Bad Request`: Returns an error if the video or userId is missing.

### `/check-face` [POST]

This endpoint is used to upload an image file for face verification against the previously created model associated with the user ID.

#### Arguments:

-   `image`: The image file of the user to be uploaded.
-   `userId`: A unique identifier for the user, which should match the ID used to create the model.

#### Responses:

-   `200 OK`: Returns success message if the image is successfully uploaded and the face is identified.
-   `400 Bad Request`: Returns an error if the image or userId is missing.
