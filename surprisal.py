import math
import string
import os
import frontmatter

from nltk.tokenize import word_tokenize
from collections import Counter
import nltk
import markdown
import bs4
from transformers import pipeline
import jinja2

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

counts = Counter()

for article in text:
    for word in article.split():
        # if not ascii, ignore
        try:
            word.encode("ascii")
        except:
            continue
        counts[word] += 1

surprisals = []
probabilities = {}

for word in counts:
    probabilities[word] = counts[word] / len(text)

    surprisals.append(-math.log(probabilities[word]))

input_prose = ""

with open("stopwords.txt") as f:
    stopwords = f.read().split("\n")

index = {}
words_with_context = []

for i, filename in enumerate(os.listdir("../_posts")):
    print(i, len(os.listdir("../_posts")))
    try:
        if not filename.endswith(".md"):
            continue

        title = frontmatter.load("../_posts/" + filename).metadata.get("title", filename)

        input_prose = frontmatter.load("../_posts/" + filename).content

        input_prose_text = input_prose

        # get plain text from content, no markdown
        input_prose = markdown.markdown(input_prose)

        # remove html tags
        input_prose = bs4.BeautifulSoup(input_prose, "html.parser").text

        # print(input_prose)

        # remove numbers
        input_prose = input_prose.translate(str.maketrans("", "", string.digits))

        # remove single quotes
        input_prose = input_prose.replace("' ", "").replace("' ", "")

        # remove punct but keep -
        input_prose = input_prose.translate(
            str.maketrans("", "", string.punctuation.replace("-", ""))
        )

        # remove --
        input_prose = input_prose.replace("--", " ")

        # encode as unicode
        input_prose = input_prose.encode("ascii", errors="ignore").decode()

        input_prose = input_prose.lower()

        # remove stopwords
        stops = stopwords

        input_prose = [word for word in input_prose.split() if word not in stops]

        # get surprisals
        surprisals = [
            -math.log(probabilities.get(word, 0.0001)) for word in input_prose
        ]

        # get 5 most surprising words
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
        mapped_words = [word for word in mapped_words if word[0] > 8]

        flat_mapped_words = [word for surprisal, word in mapped_words][:10]

        # dedupe
        flat_mapped_words = list(set(flat_mapped_words))

        # filename = i
        filename = title
        for count, word in enumerate(flat_mapped_words):
            word = word.strip()
            if index.get(word, None) is not None:
                index[word].append(filename)
                words_with_context.append([word, filename, input_prose_text])
            elif count < 5:
                index[word] = [filename]

                words_with_context.append([word, filename, input_prose_text])

        for count, ner in enumerate(all_ners):
            ner = ner.strip()
            if ner in index:
                index[ner].append(filename)
                words_with_context.append([ner, filename, input_prose_text])
            elif count < 5:
                index[ner] = [filename]

                words_with_context.append([ner, filename, input_prose_text])

    except Exception as e:
        # may happen with unicode encoding error loading the markdown files
        pass

# dedupe refs in index
for word in index:
    index[word] = list(set(index[word]))

contexts = {}

new_index = index.copy()

# if index has lowercase, merge with uppercase
for word in index:
    if word.lower() in new_index:
        new_index[word.lower()] += new_index[word]
        del new_index[word]
    elif word.lower() in stopwords:
        del new_index[word]

index = new_index

# order alphabetically
index = dict(sorted(index.items()))

import json

with open("index.json", "w") as f:
    json.dump(index, f)

dl_dds = []

# for word, filename, text in words_with_context:
#     # normalize words_with_context[w] to a big string, no new lines
#     if contexts.get(filename, None) is None:
#         contexts[filename] = {}

#     if contexts[filename].get(word, None) is None:
#         contexts[filename][word] = text.replace("\n", " ")

#     contexts[filename][word] = nlp(
#         question=f"question: what is the context of {word}, summarized like a book index?",
#         context=contexts[filename][word],
#     )["answer"]
#     print(filename, word, contexts[filename][word])

for word, filenames in index.items():
    dl = "<dl><dt>" + word + "</dt><dd><ul>"

    for filename in filenames:
        if contexts.get(word, None) is not None:
            dl += "<li>" + str(filename) + " (" + contexts[filename][word] + ")</li>"
        else:
            dl += "<li>" + str(filename) + "</li>"

    dl += "</ul>"

    dl += "</dd></dl>"

    dl_dds.append(dl)

# open index_template.html
with open("index_template.html") as f:
    template = f.read()

template = jinja2.Template(template)

template = template.render(results="\n".join(dl_dds))

# write to index.html
with open("results.html", "w") as f:
    f.write(template)