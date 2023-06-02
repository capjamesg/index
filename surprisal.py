import math
import string
import os
import frontmatter

from nltk.tokenize import word_tokenize
from collections import Counter, OrderedDict
import nltk
import json
import markdown
import bs4
from transformers import pipeline
import jinja2
import contractions
import spacy

nlp = spacy.load("en_core_web_sm")

# nlp = pipeline("question-answering")

# import sentence_transformers

# model = sentence_transformers.SentenceTransformer('bert-base-nli-mean-tokens')

# # Compute embeddings
# corpus_embeddings = model.encode(words)

# # get cosine similarity
# cos_sim = sentence_transformers.util.pytorch_cos_sim

# # Compute cosine similarity between all pairs
# cos_sim_matrix = cos_sim(corpus_embeddings, corpus_embeddings)

# # get similarity between "US" and "U.S."
# print(cos_sim_matrix[0][1])

# # get similarity between "US" and "country"
# print(cos_sim_matrix[0][2])

with open("/Users/james/Downloads/nytimes_news_articles.txt") as f:
    text = f.read()

# text is URL\n\nCONTENT
text = text.split("\n\n")

# remove URLs
text = [text for text in text if not "http" in text]

def calculate_surprisals(text: str) -> dict:
    counts = Counter()

    for article in text:
        for word in article.split():
            # if not ascii, ignore
            try:
                word.encode("ascii")
            except:
                continue
            counts[word] += 1

    return counts

print("calculating surprisals")

counts = calculate_surprisals(text)

surprisals = []
surprisals_as_dict = {}
probabilities = {}

for word in counts:
    probabilities[word] = counts[word] / len(text)

    surprisals.append(-math.log(probabilities[word]))
    surprisals_as_dict[word] = -math.log(probabilities[word])

with open("nyt_surprisals.json", "w") as f:
    json.dump(surprisals_as_dict, f)

input_prose = ""

def load_stopwords():
    with open("stopwords.txt") as f:
        stopwords = f.read().split("\n")

    return stopwords

stopwords = load_stopwords()

index = {}
words_with_context = []

lemmatize = nltk.stem.WordNetLemmatizer()

def process_prose(input_prose, title, url):
    input_prose_text = input_prose

    # get plain text from content, no markdown
    input_prose = markdown.markdown(input_prose)

    # get text only from p tags
    input_prose = " ".join([p.text for p in bs4.BeautifulSoup(input_prose, "html.parser").find_all("p")])

    # remove all urls
    input_prose = " ".join([word for word in input_prose.split() if not "http" in word])

    tokenize = nlp(input_prose)

    # remove non-unicode
    tokenize = [token for token in tokenize if token.is_ascii]

    tokenize = [token for token in tokenize if not token.text[0] in string.punctuation and not token.text[-1] in string.punctuation]

    # get strings
    input_prose = " ".join([token.text for token in tokenize if not token.is_stop and not token.is_punct])

    # contract
    input_prose = contractions.fix(input_prose)

    # # remove numbers
    # input_prose = input_prose.translate(str.maketrans("", "", string.digits))

    # # remove punct but keep - and ., as long as they're not at the end of a word
    # input_prose = input_prose.translate(
    #     str.maketrans("", "", string.punctuation.replace("-", "").replace(".", ""))
    # )

    # # remove --
    # input_prose = input_prose.replace("--", " ")

    # # encode as unicode
    # input_prose = input_prose.encode("ascii", errors="ignore").decode().lower()

    input_prose = [word.strip().strip(".") for word in input_prose.split() if word not in stopwords]

    # get surprisals
    surprisals = [
        -math.log(probabilities.get(word, 0.0001)) for word in input_prose
    ]

    mapped_words = list(zip(surprisals, input_prose))

    mapped_words.sort()
    mapped_words.reverse()

    ners = nltk.ne_chunk(nltk.pos_tag(word_tokenize(input_prose_text)))

    # only get proper nouns
    ners = [ner for ner in ners if isinstance(ner, nltk.tree.Tree)]

    all_ners = []

    for ner in ners:
        all_ners.append(" ".join([word for word, tag in ner.leaves()]))

    # index should have format
    # {word: [surprisal, count]}
    # calc index on input

    # remove if surprisal < 8
    mapped_words = [word for word in mapped_words if word[0] > 7]

    flat_mapped_words = [word for surprisal, word in mapped_words][:10]
    flat_mapped_words = list(set(flat_mapped_words))

    print(flat_mapped_words)

    # filename = i
    filename = title

    for count, word in enumerate(flat_mapped_words):
        word = word.strip()
        if index.get(word, None) is not None:
            index[word].append((filename, url))
            words_with_context.append([word, filename, input_prose_text, url])
        elif count < 5:
            index[word] = [(filename, url)]

            words_with_context.append([word, filename, input_prose_text, url])

    for count, ner in enumerate(all_ners):
        ner = ner.strip()
        if ner in index:
            index[ner].append((filename, url))
            words_with_context.append([ner, filename, input_prose_text, url])
        elif count < 5:
            index[ner] = [(filename, url)]

            words_with_context.append([ner, filename, input_prose_text, url])

for i, filename in enumerate(os.listdir("../_posts")):
    print(i, len(os.listdir("../_posts")))
    try:
        if not filename.endswith(".md"):
            continue

        file_contents = frontmatter.load("../_posts/" + filename)

        title = file_contents.metadata.get("title", filename)

        input_prose = file_contents.content

        # map filename to url

        # date is yyyy-mm-dd in filename
        date = filename.split("-")[:3]

        # slug is everything after yyyy-mm-dd-
        slug = filename.split("-")[3:]

        date = "/".join(date)

        url = "https://jamesg.blog/" + date + "/" + "-".join(slug).replace(".md", "") + "/"

        print(url)

        process_prose(input_prose, title, url)

    except Exception as e:
        # raise e
        # may happen with unicode encoding error loading the markdown files
        pass

# dedupe refs in index
for word in index:
    index[word] = list(set(index[word]))

contexts = {}

new_index = index.copy()

# if index has lowercase, merge with uppercase
# ensure punctuation is removed for merging
for word in index:
    # remove puncts
    new_word = word.translate(str.maketrans("", "", string.punctuation)).lower()
    if new_word in new_index:
        new_index[new_word] += new_index[new_word]
        del new_index[new_word]
    elif word in stopwords:
        del new_index[word]

index = new_index

# order lexicographically
index = OrderedDict(sorted(index.items()))

new_index = index.copy()

# merge lower with upper cases
for word in index:
    # remove puncts
    new_word = word.translate(str.maketrans("", "", string.punctuation)).lower()
    if new_word in new_index:
        new_index[new_word] += new_index[new_word]
        del new_index[new_word]
    elif word in stopwords:
        del new_index[word]

index = new_index

letters_in_index = []

for word in index:
    letters_in_index.append(word[0])

letters_in_index = list(set(letters_in_index))

# order alphabetically by first letter in key
index = OrderedDict(sorted(index.items()))

import json

with open("index.json", "w") as f:
    json.dump(index, f)

dl_dds = []

letter_headings = set()

for word, filenames in index.items():
    if word[0][0] not in letter_headings:
        letter_headings.add(word[0][0])

        dl_dds.append("<h2 id='" + word[0][0].upper() + "'>" + word[0][0].upper() + "</h2>")

    dl = "<dl><dt>" + word + "</dt><dd><ul>"

    for filename, url in filenames:
        if contexts.get(word, None) is not None:
            dl += "<li><a href='" + url + "'>" + str(filename) + "</a>: " + contexts[word][filename] + "</li>"
        else:
            dl += "<li><a href='" + url + "'>" + str(filename) + "</a></li>"

    dl += "</ul>"

    dl += "</dd></dl>"

    dl_dds.append(dl)

# open index_template.html
with open("index_template.html") as f:
    template = f.read()

template = jinja2.Template(template)

template = template.render(results="\n".join(dl_dds), letters_in_index=letters_in_index)

with open("index.html", "w") as f:
    f.write(template)