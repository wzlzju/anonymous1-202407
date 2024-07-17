import json

import flask
import networkx as nx




app = flask.Flask(__name__, static_folder="force")


@app.route("/")
def static_proxy():
    return app.send_static_file("force.html")


app.run(port=8000)