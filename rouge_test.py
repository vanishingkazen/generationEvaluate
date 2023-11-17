"""
返回一个字典，在三个rouge标准下的 p：查准率  R：召回率  F1值
"""

#该使用哪个指标
# coding:utf8

from rouge import Rouge
def getRougescore(reference_sentences,generated_sentences,rouge_n='l'):

    rouge = Rouge()
    rouge_score = rouge.get_scores(generated_sentences, reference_sentences)
    return rouge_score[0]["rouge-{}".format(rouge_n)]['f']

if __name__ == '__main__':
    generated_sentences = ["我 是 一 个 男 孩"]
    reference_sentences = ["你 是 一 个 男 孩"]
    print(getRougescore(reference_sentences,generated_sentences))


