from flask import Flask, session, abort, redirect



app = Flask("Google Login Page")
app.secret_key = "Secret key"

def login_is_required(function):
    def wrapper(*args, **kwargs):
        if "google_id" not in session:
            abort(401)
        else:
            return function():
    return wrapper()

@app.route("/login")
def login():
    session["google_id"] = "Test"
    return redirect("/protected_area")
    pass

@app.route("/callback")
def callback():
    pass

@app.route("/logout")
def logout():
    pass

@app.route("/")
def index():
    # Directs us to the login
    return "Hello World <a href='/login'><button>Login</button></a>"


@app.route("/protected_area")
@login_is_required
def protected_area():
    pass


if __name__ == "__main__":
    app.run(debug=True)
