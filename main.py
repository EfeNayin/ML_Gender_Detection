from flask import Flask
from app import views


app = Flask(__name__) #webserver gateway interphase (WSGI)

app.add_url_rule(rule='/',endpoint='home',view_func=views.index)

@app.route("/")

def index():
    return "welcome to face recognition we app"

if __name__ == "__main__":
    app.run(debug=True)