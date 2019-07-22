from flask import Flask, request, jsonify, render_template
import time
from helper import labeler, prediction
import os

app = Flask(__name__)

port = int(os.environ.get("PORT", 5000))


@app.route('/')
def index():
    return render_template("home.html")

@app.route('/stats/')
def stats():
    return render_template("statistics.html")

@app.route('/api/')
def predict():
    if ('url' in request.args) and (request.args['url'] != ''):
        url = request.args['url']
    else:
        return jsonify({ 'res' : 'Error' })

    labelData = prediction(url)
    if(len(labelData)>0):
        res = list(labelData[0])
        css = labelData[1]
        print(res, css)
        return jsonify({'res': res, 'cssClass': css})
    else:
        return jsonify({ 'res' : 'Error' })

if(__name__==  "__main__"):
    app.run(host='0.0.0.0', port=port, debug=True)