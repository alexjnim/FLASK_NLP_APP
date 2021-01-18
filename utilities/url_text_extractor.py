from newspaper import Article
import validators


def url_text_extractor(url):
    """
    Input
    ----------
    url (string)
    Returns
    ----------
    An article's text
    """
    url = url.strip()
    article = Article(url)
    article.download()
    article.parse()

    return article.text


def check_for_url(text):
    """
    Check if text input is a url. If so it extracts the website text
    else it uses the input text.
    Input
    ----------
    text (string): Text or a url.
    Returns
    ----------
    An article's text
    """
    text = text.strip()
    valid = validators.url(text)

    if valid is True:
        text = url_text_extractor(text)
    else:
        pass

    # clean the text
    text = text.replace("\n", " ").replace("\r", "").replace("  ", " ")

    return text


# -
def get_url_title(request):
    """
    Get the title of the article if the link is a URL, starting from the Flask request variable.
    Input
    ----------
    request: flask request variable
    Returns
    ----------
    An article's title
    """

    url = request.form["text"]
    url = url.strip()

    valid = validators.url(url)
    title = "not title"
    if valid is True:
        article = Article(url)
        article.download()
        article.parse()
        title = article.title
    else:
        pass
    return title
