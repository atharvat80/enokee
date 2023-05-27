import csv
import json
from collections import Counter
from tqdm.auto import tqdm
from zipfile import ZipFile


ent2idx = {}
with open('entities.json', 'r') as infile:
    entities = json.load(infile)
    ent2idx = {v['pageid']: int(k) for k, v in entities.items()}


def generate_dataset(f_path, small=True, max_mentions=500):
    def process(inp_file, total, num_sentences):
        for article in inp_file:
            article = json.loads(article)
            # for each paragraph
            for p in article['paras_info']:
                # for each sentence in paragraph
                for sent in p:
                    total += 1
                    # if a sentence too short ot too long skip it
                    if len(sent['context']) < 2 or len(sent['context']) > 2500:
                        continue
                    # record entry
                    entry = {'context': sent['context'],
                             'spans': [], 'targets': []}
                    for target, start, end, name, _ in sent['a_list']:
                        if target in ent2idx.keys():
                            entry['spans'].append((start, end))
                            entry['targets'].append(ent2idx[target])
                    # if a sentence has not entity mentions, skip it
                    if not entry['targets']:
                        continue
                    # only keep a sentence if any entity is below max # mentions
                    if small:
                        keep = any([num_mentions[i] <= max_mentions 
                                    for i in entry['targets']])
                    else:
                        keep = True
                    
                    if keep:
                        writer.writerow(entry)
                        for i in entry['targets']:
                            num_mentions[i] += 1
                        num_sentences += 1
        
        return total, num_sentences

    total, num_sentences = 0, 0
    num_mentions = Counter()
    fieldnames = ['context', 'spans', 'targets']
    outname = "PELT-corpus-small.csv" if small else "PELT-corpus.csv"
    outfile = open(outname, "w", newline='', encoding='utf-8')
    writer = csv.DictWriter(outfile, delimiter=',', fieldnames=fieldnames,
                            quoting=csv.QUOTE_NONNUMERIC)
    with ZipFile(f_path, mode="r") as archive:
        files = archive.namelist()
        p_bar = tqdm(files, desc="Processing files")
        for file in p_bar:
            with archive.open(file, 'r') as in_file:
                total, num_sentences = process(in_file, total, num_sentences)
                p_bar.set_postfix({'total': total, 'added': num_sentences})

    outfile.close()

    print("Done.")
    print("Most common entities & their # of mentions:")
    print(*list(num_mentions.most_common())[:5], sep="\n")
    print("Least common entities & their # of mentions:")
    print(*list(num_mentions.most_common())[-5:], sep="\n")
    print("Total number of mentions:", num_mentions.total())
    print("Avg. # mentions / entity:", num_mentions.total()//len(ent2idx.keys()))


generate_dataset("articles.zip", small=False)
