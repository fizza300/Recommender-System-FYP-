from flask import Flask, render_template, redirect

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.hbs')

@app.route('/pf')
def preference():
    return redirect("http://localhost:8502", code=302)

if __name__ == "__main__":
    app.run(debug=True)
