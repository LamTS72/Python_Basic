import numpy as np
import os
import urllib.request
from bs4 import BeautifulSoup


def url2text(url):
    """
    Reference:
    * https://stackoverflow.com/questions/45768441/hangs-on-open-url-with-urllib-python3
    """
    
    req = urllib.request.Request(
        url, 
        data=None, 
        headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
            }
    )
    html = urllib.request.urlopen(req)
    soup = BeautifulSoup(html, features="html.parser")
    
    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out
    # get text
    text = soup.get_text()
    
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text

links = {
    "https://www.bbc.com/news/business-67304260": "business",
    "https://www.bbc.com/news/business-67305453": "business",
    "https://www.bbc.com/sport/football/67323707": "sport",
    "https://www.bbc.com/sport/football/67324261": "sport",
    "https://www.bbc.com/news/technology-67302788": "tech",
    "https://www.bbc.com/news/science-environment-67243772": "tech",
    "https://www.bbc.com/news/entertainment-arts-67207695": "entertainment",
    "https://www.bbc.com/news/entertainment-arts-67311637": "entertainment",
    "https://www.bbc.com/news/uk-politics-67320861": "politics",
    "https://www.bbc.com/news/world-europe-67321777": "politics"
}

def link2text(links):
    contents = [(url2text(url), links[url] ) for url in links.keys()]
    text = [c[0] for c in contents]
    labels = [c[1] for c in contents]
    return text, labels
    