#encoding=utf-8

import gensim

model = gensim.models.Word2Vec.load("model/wiki.ch.text.model")
result = model.most_similar(u'第一夫人')
print model['ming']
print "start"
for e in result:
    print e[0],e[1]
print "end"
