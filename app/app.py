from flask import Flask, render_template, request, redirect, url_for
import model

app = Flask(__name__)

@app.route("/")
def admin():
    return render_template("admin.html")

@app.route("/show")
def show():
    return render_template("show.html", query_show=model.get_query())

@app.route("/postentry", methods=["POST"])
def postentry():
    query = request.form["query"]
    model.add_query(query)
    return redirect(url_for('show'))

@app.route("/logout")
def logout():
    model.destroy()
    return redirect("/")

if __name__=="__main__":
    model.init()
    app.run(debug=True)