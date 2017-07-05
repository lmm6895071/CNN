#encoding:utf-8

import jieba
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

jieba.enable_parallel(4)
infile = open("true_neg.txt")
outfile = file("cut_neg.txt","w")
for line in infile.readlines():
    outfile.write(" ".join(jieba.cut(line)))
outfile.close()
infile.close()
jieba.disable_parallel()



