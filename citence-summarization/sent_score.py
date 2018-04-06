# -*-coding:utf-8-*-

from util.pos_tagging import get_nouns
from util.file_op import read_file_utf8, write_file_utf8
from util.abs_paths import get_paths
from get_tfidf_features import tfidf_features
from nltk import sent_tokenize, word_tokenize
import numpy as np
import gmpy2
import operator

paths = get_paths('./data')
fid = 20
print '[',fid,']', paths[fid]
lines,_, summary = read_file_utf8(paths[fid]).split(u'---')
lines = [' '.join(l.split('\t')[2:4]) for l in lines.split('\n') if len(l)>1]
score_dic = {'#n':0, '#TFIDF_words':0, 'senti':0, 'ti':0, 'target_pos':0}
cite_dic = dict([str(i),dict.copy(score_dic)] for i in range(len(lines)))


# 접속 부사 - 대조
conj_adv = [l.split('\t')[0] for l in read_file_utf8('util/conj_advs.txt').split('\n') if l.count(u'대조')>0]
# print conj_adv
max_cite_len = 0
min_cite_len = 200
for i in range(len(lines)):
    sent = lines[i].split()
    max_cite_len = max(max_cite_len, len(sent))
    min_cite_len = min(min_cite_len, len(sent))
    key = str(i)
    if sent[-1]=='POS' or sent[-1]=='NEG':
        cite_dic[key]['senti'] = round(10.0/len(sent),4)
    else:
        cite_dic[key]['senti'] = round(1.0/len(sent),4)

    ti = float(sent.index('TARGET'))
    d = 0
    for adv in conj_adv:
        if sent.count(adv)>0:
            # print adv, ti, sent.index(adv)
            # print lines[i]
            d = max(d, round(abs(len(sent)/float(sent.index(adv) - ti)), 4) )#/ len(sent

    if d>0:
        # print d
        cite_dic[key]['target_pos'] = round(np.log2(float(gmpy2.root(d, len(sent)))), 4)
# print cite_dic['4']
    # cite_dic[str(i)] = sum(cite_dic[str(i)].values()) # SUM
# cite_dic = sorted(cite_dic.items(), key=itemgetter(1)) #soring
print 'max-cite-len', max_cite_len
print 'min-cite-len', min_cite_len

print '________________________'
data = [' '.join(line.split()[:-1]) for line in lines]
features = tfidf_features(data, 0.2)
sents = []
sent_dic = {}

for i,val in sorted(cite_dic.items()):
    cite_sents = sent_tokenize(data[int(i)])
    sents += cite_sents

    for j in range(len(cite_sents)):
        # print i, val, len(sent_dic)
        sent_dic[str(len(sent_dic))] = val


for i, s in sent_dic.items():
    idx = int(i)
    wn = 0.0

    sent_words = word_tokenize(sents[idx])
    for w in sent_words:
        if features.count(w)>0:
            wn += features.count(w)
    wn = wn/len(sent_words)
    sent_dic[i]['#TFIDF_words'] = round(wn, 4)

    nn = round(float(len(get_nouns(sents[idx]))) / float(len(sent_words)), 4)
    sent_dic[i]['#n'] = round(nn, 4)

    if sent_words.count('TARGET')>0:
        sent_dic[i]['ti'] = 1.0
    else:
        sent_dic[i]['ti'] = 0.1

    sent_dic[i]['ti'] = round(sent_dic[i]['ti']/len(sent_words), 4)

    sent_dic[i] = sum([val for key, val in sent_dic[i].items()])
    # print i, sent_dic[i], s


sorted_sent_dic = sorted(sent_dic.items(), key=operator.itemgetter(1), reverse=True)

selected_sents = [feat for feat, val in sorted_sent_dic[:int(len(sents)*0.3)]]
selected_sents = [sents[int(idx)] for idx in selected_sents]

write_file_utf8(paths[fid].replace('data', 'output'), '\n'.join(selected_sents))
