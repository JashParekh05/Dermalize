from flask import Flask

app = Flask("Google Login Page")
app.secret_key = "Secret key"

@app.route("/login")
def login():
    pass

@app.route("/callback")
def callback():
    pass

@app.route("/logout")
def logout():
    pass

@app.route("/")
def index():
    return "Hello World"

@app.route("/protected_area")
def protected_area():
    pass


if __name__ == "__main__":
    app.run(debug=True)
