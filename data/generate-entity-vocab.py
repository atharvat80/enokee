import json
import requests
from collections import Counter
from zipfile import ZipFile
from tqdm.auto import tqdm
from urllib.parse import unquote

# from categories import CATEGORIES
from zelda import TEST_ENTITIES

def count_mentions(f_path):
    def count(inp_file):
        for article in inp_file:
            article = json.loads(article.decode())
            # for each paragraph
            for p in article['paras_info']:
                # for each sentence in paragraph
                for sent in p:
                    # count number of mentions
                    for a in sent['a_list']:
                        num_mentions[a[0]] += 1

    num_mentions = Counter()
    with ZipFile(f_path, mode="r") as archive:
        files = archive.namelist()
        for file in tqdm(files, desc="Reading Corpus".ljust(25)):
            with archive.open(file, 'r') as infile:
                count(infile)

    return num_mentions


def get_article_title_revid(pageids: int, session = None) -> int:
    """
    Get title of Wikipedia article
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "rvslots":"*",
        "rvprop":"ids",
        "pageids": '|'.join(pageids) if isinstance(pageids, list) else pageids,
    }
    
    if session:
        r = session.get(url=url, params=params)
    else:
        r = requests.get(url=url, params=params)
    
    r = r.json()
    return r['query']['pages']


def get_predicted_category(revids, session = None) -> list[str]:
    """
    Get predicted category of article using it's revision id
    """
    url = "https://ores.wikimedia.org/v3/scores/enwiki/?"
    params = {
        "models": "articletopic",
        "revids": '|'.join(revids) if isinstance(revids, list) else revids
    }
    
    if session:
        r = session.get(url=url, params=params)
    else:
        r = requests.get(url=url, params=params)
    
    r = r.json()
    scores = r['enwiki']['scores']
    return scores


if __name__ == "__main__":
    # count mentions
    num_mentions = count_mentions("articles.zip")

    # get entity titles and revids
    entities = []
    num_entities = 50000

    # get popular + test entities
    ordered_mentions = [(ent, num_mentions[ent]) for ent in TEST_ENTITIES 
                        if num_mentions[ent] > 0]
    print(len(TEST_ENTITIES) - len(ordered_mentions), 
          "of test entities with no mentions in the dataset") 
    for ent, mention in num_mentions.most_common():
        if ent not in TEST_ENTITIES:
            ordered_mentions.append((ent, mention))
        if len(ordered_mentions) > (num_entities + 10000):
            break

    # get entity titles and revids
    idx = 0
    session = requests.session()
    pbar = tqdm(total=num_entities, desc="Generating Entity Vocab".ljust(25))

    while len(entities) < num_entities:
        ent_ids = [str(id_) for (id_, _) in ordered_mentions[idx: idx+50]]
        ent_info = get_article_title_revid(ent_ids, session)
        for ent in ent_info.values():
            try:
                pageid = ent['pageid']
                title = unquote(ent['title'])
                # revid = ent['revisions'][0]['revid']
            except KeyError:
                if ent['pageid'] in TEST_ENTITIES:
                    print("Error getting test entity", ent)
            else:
                if len(entities) < num_entities:
                    entities.append(dict(pageid=pageid, title=title)) #, revid=revid))
                    pbar.update()

        idx += 50

    # sort by count and assign index
    entities.sort(key = lambda ent: num_mentions[ent['pageid']], reverse=True)
    entities = [dict(pageid=0, title='[MASK]')] + entities
    entities = {idx: ent for idx, ent in enumerate(entities)}
    
    # Getting categories
    # for idx in tqdm(range(0, num_entities, 50), desc="Getting Categories".ljust(25)):
    #     revid2id = {}
    #     for i in range(idx, idx+50):
    #         revid2id[str(entities[i]['revid'])] = i

    #     if not revid2id:
    #         continue
        
    #     scores = get_predicted_category(list(revid2id.keys()), session)
    #     preds = {}
    #     for id, res in scores.items():
    #         idx = revid2id[id]
    #         try:
    #             cats = res['articletopic']['score']['prediction']
    #         except KeyError:
    #             probs = list(res['articletopic']['score']['probability'].items())
    #             probs.sort(key=lambda x: x[1], reverse=True)
    #             cats = [i[0] for i in probs[:1]]

    #         cats = set([j.replace('*', '') for i in cats for j in i.split('.')])
    #         cats = list(cats.intersection(CATEGORIES))
    #         entities[idx]['cats'] = cats 
    #         del entities[idx]['revid']

    session.close()
    # save as json
    with open("entities.json", "w") as f:
        json.dump(entities, f)
