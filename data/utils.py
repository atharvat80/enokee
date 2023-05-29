import signal
from urllib.parse import unquote

import requests
from bs4 import BeautifulSoup


def get_article_latest_revid(title: str, session=None) -> int:
    """
    Get latest revision id of Wikipedia article
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "rvslots": "*",
        "rvprop": "ids",
        "titles": title,
    }
    if session:
        r = session.get(url=url, params=params)
    else:
        r = requests.get(url=url, params=params)
    r = r.json()
    try:
        r = r["query"]["pages"].values()
        return list(r)[0]["revisions"][0]["revid"]
    except Exception:
        return


def get_predicted_category(revid: int, top_k=3, threshold=0.6) -> list[str]:
    """
    Get predicted category of article using it's revision id
    """
    url = "https://ores.wikimedia.org/v3/scores/enwiki/?"
    params = {"models": "articletopic", "revids": revid}
    r = requests.get(url=url, params=params)
    r = r.json()
    scores = r["enwiki"]["scores"][str(revid)]["articletopic"]["score"]
    probs = list(scores["probability"].items())
    probs.sort(key=lambda x: x[1], reverse=True)
    return [cat for cat, prob in probs[:top_k] if prob > threshold]


def get_article_text(title: str, session=None, verbose=0) -> str:
    """
    Get HTML contents of a Wikipedia article
    """
    title = title.replace(" ", "_")
    url = "https://en.wikipedia.org/wiki/" + requests.compat.quote_plus(title)
    res = session.get(url) if session else requests.get(url)
    if res.status_code == requests.codes.ok:
        return unquote(res.text)
    else:
        if verbose:
            print("Got status code", res.status_code, "getting", url)


def parse_paragraph_anchors(text: str, ent2idx: dict[str, int] = {}) -> dict:
    """
    Get anchor offsets, targets and anchor text lengths in a paragraph
    """
    targets, lengths, offsets = [], [], []
    out = ""
    idx = 0
    while idx < len(text):
        if text[idx] == "<":
            # find the end of the anchor tag
            end = idx + text[idx:].find("</a>") + 4
            # get the href and anchor text
            anchor = BeautifulSoup(text[idx:end], "lxml").a
            entity = anchor["href"][6:].replace("_", " ")
            offsets.append(len(out))
            lengths.append(len(anchor.text.strip()))
            targets.append(ent2idx[entity] if ent2idx else entity)
            out += anchor.text.strip()
            idx = end
        else:
            out += text[idx]
            idx += 1

    return {
        "text": out.rstrip(),
        "targets": targets,
        "lengths": lengths,
        "offsets": offsets,
    }


def parse_article_text(text: str, ent2idx: dict[str, int] = {}) -> dict:
    """
    Parse HTML of a Wikipedia Article

    Args
    ---
    text
        HTML string of the article

    entity_vocab (optional)
        a dictionary of the entities to keep, all anchors will be kept if None


    Returns
    ---
    list (optional)
        a list of dict containing
        {
            'anchors': list[str],
            'offsets': list[int],
            'targets': list[str],
            'text': str
        }
    """
    parsed = []
    soup = BeautifulSoup(text, "lxml").body

    # remove citation text and anchors
    for elem in soup.find_all("sup"):
        elem.decompose()

    # select all paragraphs remove all tags except anchors
    for p in soup.find_all("p", {"class": None, "style": None}):
        for e in p.find_all():
            # if the element is a wiki anchor keep it
            if e.name == "a" and e.has_attr("href") and e["href"].startswith("/wiki/"):
                entity = e["href"][6:].replace("_", " ")
                # if ent2idx is provided and the anchor is OOV entity
                # remove the anhor but keep the anchor text
                if ent2idx and entity not in ent2idx.keys():
                    e.unwrap()
            # if the element is not an anchor or plaintext remove the
            # but keep the innerHTML
            elif e.name is not None:
                e.unwrap()
        # parse paragraph html containing the anchor tags
        p = str(p)[3:-4]
        p = p.replace("\xa0", " ").replace("\n", " ")
        try:
            p = parse_paragraph_anchors(p, ent2idx)
        except Exception as e:
            pass
        else:
            if len(p["text"]) > 1:
                parsed.append(p)

    return parsed


class InterruptHandler(object):
    """
    Detect Keyboard Interrupt and define custom handler
    source: https://stackoverflow.com/questions/1112343/how-do-i-capture-sigint-in-python
    """

    def __init__(self, sig=signal.SIGINT):
        self.sig = sig

    def __enter__(self):
        self.interrupted = False
        self.released = False
        self.original_handler = signal.getsignal(self.sig)

        def handler(signum, frame):
            self.release()
            self.interrupted = True

        signal.signal(self.sig, handler)

        return self

    def __exit__(self, type, value, tb):
        self.release()

    def release(self):
        if self.released:
            return False

        signal.signal(self.sig, self.original_handler)

        self.released = True

        return True
