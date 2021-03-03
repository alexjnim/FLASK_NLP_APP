from resources.predict_category import predict_category
from resources.summarize_text import summarize_text
from resources.predict_sentiment import predict_sentiment
from resources.get_answers_from_query import get_answers_from_query
from resources.keywords_extractor import get_key_words
from utilities.url_text_extractor import get_url_title
from flask import Flask, render_template, request, json

# +
# Initialise the Flask app
app = Flask(__name__, template_folder="templates")

# Set up the main route
@app.route("/", methods=["GET", "POST"])
def main():
    if request.method == "GET":
        # Just render the initial form, to get input
        return render_template("index.html")

    if request.method == "POST":
        if len(request.form["text"]) == 0:
            error_message = "*please enter some text"
            return render_template("index.html", error_message=error_message)

        title = get_url_title(request)
        category = predict_category(request)
        summary = summarize_text(request)
        sentiment = predict_sentiment(request)
        key_words = get_key_words(request)

    return render_template(
        "results.html",
        title=title,
        category=category,
        summary=summary,
        sentiment=sentiment,
        key_words=key_words,
        text=request.form["text"],
    )


@app.route("/results.html", methods=["POST"])
def answer_query():
    if request.method == "POST":
        title = get_url_title(request)
        summary = summarize_text(request)
        if len(request.form["query"]) == 0:
            error_message = "*please enter a question or try again with another article"
            return render_template(
                "results.html",
                error_message=error_message,
                summary=summary,
                text=request.form["text"],
            )

        answers = get_answers_from_query(request)
    return render_template(
        "results.html",
        answers=answers,
        title=title,
        summary=summary,
        text=request.form["text"],
        question=request.form["query"],
    )


# -
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
