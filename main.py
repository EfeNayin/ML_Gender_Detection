import os

from flask import Flask

from app import views

app = Flask(__name__)

# Uploaded file size limit (8 MB). If exceeded, Flask returns a 413 error.
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = os.path.join("static", "upload")

# Single route: both displays the form (GET) and handles the upload (POST).
app.add_url_rule(
    rule="/",
    endpoint="home",
    view_func=views.index,
    methods=["GET", "POST"],
)

if __name__ == "__main__":
    app.run(debug=True)