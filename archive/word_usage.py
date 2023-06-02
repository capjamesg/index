import string
import os
import frontmatter
import markdown
import matplotlib.pyplot as plt

words_by_month = {}


for i, filename in enumerate(os.listdir("../_posts")):
    # print(i, len(os.listdir("../_posts")))
    try:
        if not filename.endswith(".md"):
            continue

        input_prose = frontmatter.load("../_posts/" + filename).content

        key = filename.split("-")[:2]

        words = markdown.markdown(input_prose)

        key = "-".join(key).strip()

        if words_by_month.get(key, None) is None:
            words_by_month[key] = {}

        for word in words.split():
            word = word.translate(str.maketrans("", "", string.punctuation)).lower().strip()
            words_by_month[key][word] = words_by_month[key].get(word, 0) + 1
    except:
        pass

mentions = []
dates = []

word = "joy"

# order words by month
words_by_month = dict(sorted(words_by_month.items()))

for month, words in words_by_month.items():
    mentions.append(words.get(word, 0))
    dates.append(month)

# bar
plt.title(f"Mentions of '{word}' by Month")
plt.bar(range(len(dates)), mentions, align="center")
plt.xticks(range(len(dates)), dates, rotation=90)
plt.ylabel("Mentions")
plt.xlabel("Month")
# make big
plt.gcf().set_size_inches(18.5, 10.5)

# save
plt.savefig(f"{word}.png")