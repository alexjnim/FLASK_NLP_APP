from resources.predict_category import predict_category
from resources.summarize_text import summarize_text
from resources.predict_sentiment import predict_sentiment
from flask import Flask, render_template, request, json

# +
# Initialise the Flask app
app = Flask(__name__, template_folder='templates')

# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        # Just render the initial form, to get input
        return(render_template('index.html'))

    if request.method == 'POST':

        if len(request.form['text']) == 0:
            message = '*please enter some text'
            return render_template('index.html', error_message=message)

        category = predict_category(request)
        summary = summarize_text(request)
        sentiment = predict_sentiment(request)
        return render_template('results.html', category = category, summary = summary, sentiment = sentiment)

# -
if __name__ == "__main__":
    app.run(port=5000, host='0.0.0.0')
