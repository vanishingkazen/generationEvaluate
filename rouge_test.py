"""
返回一个字典，在三个rouge标准下的 p：查准率  R：召回率  F1值
"""
# coding:utf8
from rouge import Rouge
def getRougescore(reference_sentences,generated_sentences):
    rouge = Rouge()
    rouge_score = rouge.get_scores(generated_sentences, reference_sentences)
    return rouge_score

if __name__ == '__main__':
    generated_sentences = ["i am a student from china"]  # 预测摘要 （可以是列表也可以是句子）
    reference_sentences = ["i am a student from china"]  # 真实摘要
    print(getRougescore(reference_sentences,generated_sentences))


