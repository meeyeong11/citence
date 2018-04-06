# -*-coding:utf-8-*-

from rouge import Rouge
from util.abs_paths import get_paths
from util.file_op import read_file_utf8
from util.pos_tagging import tagging_no_tag

folder = './output/'
files = get_paths(folder)
scores1 = {'p':0,'r':0, 'f':0}
scores2 = {'p':0,'r':0, 'f':0}
scoresl = {'p':0,'r':0, 'f':0}
maxf1 = 0
maxf2 = 0
maxfl = 0
for f in files:
    print f
    h = read_file_utf8(f).replace('\n','')
    gold_f = f.replace('output','gold_summary')

    r = tagging_no_tag(read_file_utf8(gold_f).replace('\n',''))
    h = tagging_no_tag(h)

    sorted_vocab = dict([(w, str(i)) for i, w in enumerate(list(sorted(set(r+h))))])

    h = ' '.join( [sorted_vocab[w] for w in h])
    r = ' '.join([sorted_vocab[w] for w in r])

    rouge = Rouge()
    scores = rouge.get_scores(h,r)[0]

    scores1['p'] += scores['rouge-1']['p']
    scores1['r'] += scores['rouge-1']['r']
    scores1['f'] += scores['rouge-1']['f']
    maxf1 = max(scores['rouge-1']['f'], maxf1)

    scores2['p'] += scores['rouge-2']['p']
    scores2['r'] += scores['rouge-2']['r']
    scores2['f'] += scores['rouge-2']['f']
    maxf2 = max(scores['rouge-2']['f'], maxf2)

    scoresl['p'] += scores['rouge-l']['p']
    scoresl['r'] += scores['rouge-l']['r']
    scoresl['f'] += scores['rouge-l']['f']
    maxfl = max(scores['rouge-l']['f'], maxfl)

print 'rouge-1', 'p :', round(scores1['p']/len(files),4), 'r :', round(scores1['r']/len(files),4), 'f :', round(scores1['f']/len(files),4)
print 'rouge-2', 'p :', round(scores2['p']/len(files),4), 'r :', round(scores2['r']/len(files),4), 'f :', round(scores2['f']/len(files),4)
print 'rouge-l', 'p :', round(scoresl['p']/len(files),4), 'r :', round(scoresl['r']/len(files),4), 'f :', round(scoresl['f']/len(files),4)
print 'max rouge-1 f score: ', round(maxf1,4)
print 'max rouge-2 f score: ', round(maxf2,4)
print 'max rouge-l f score: ', round(maxfl,4)