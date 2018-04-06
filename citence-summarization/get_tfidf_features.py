# -*-coding:utf-8-*-


def tfidf_features(sents, rate):

    # sents: 공백으로 나눠진 sentence (string type)
    # rate: tfidf 상위 rate %
    from sklearn.feature_extraction.text import TfidfVectorizer
    from operator import itemgetter

    vectorizer = TfidfVectorizer(min_df=1)
    tfidf_model = vectorizer.fit_transform(sents)
    vocabs = vectorizer.get_feature_names()
    final_tfidf = {}
    for i in range(len(sents)):
        vocab_index = tfidf_model[i, :].nonzero()[1]
        tfidf_scores = zip(vocab_index, [tfidf_model[i, x] for x in vocab_index])
        for w, s in [(vocabs[i], round(s, 4)) for (i, s) in tfidf_scores]:
            if final_tfidf.has_key(w):
                final_tfidf[w] = max(s, final_tfidf[w])

            else:
                final_tfidf[w] = s

    final_tfidf = list(sorted(final_tfidf.items(), key=itemgetter(1), reverse=True))
    print len(final_tfidf), int(len(final_tfidf)*rate)
    # for w,s in final_tfidf:
    #     print w, s
    return [feat for feat, val in final_tfidf[:int(len(final_tfidf)*rate)]]