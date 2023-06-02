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