from flask import Flask 

app = Flask(__name__) #webserver gateway interphase (WSGI)

@app.route("/")

def index():
    return "welcome to face recognition we app"

if __name__ == "__main__":
    app.run(debug=True)