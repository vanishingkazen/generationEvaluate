'''
:param reference_sentence str类型，只能传一个句子，如 "This is the reference sentence."
:param generated_sentence str类型，只能传一个句子，如 "This is the generated sentence."
:param n_gram_ls list, 存储几个要评估的 n_gram, 如 [1,2] 函数会返回在 n_gram=1和n_gram=2的情况下 评价的分数

:return dictionary{
    'precision':精确率,
    'recall':召回率,
    'fscore':综合分数,
    'tp':true positive,
    'n_gram'：n_gram
}
'''
from nltk.translate.chrf_score import chrf_precision_recall_fscore_support

# precision准确率 , recall 召回率, fscore, tp(true positive)
def get_chrf_precision_recall_fscore_support(reference_sentence,generated_sentence,n_gram_ls=[1]):
    result = []
    for n_gram in n_gram_ls:
        precision, recall, fscore, tp = chrf_precision_recall_fscore_support(
            reference_sentence.split(), generated_sentence.split(),n=n_gram,epsilon=0.
        )
        result.append({
            'precision':precision,
            'recall':recall,
            'fscore':fscore,
            'tp':tp,
            'n_gram':n_gram
        })
    return result

if __name__ == '__main__':
    reference_sentence = "This is the reference sentence."
    generated_sentence = "This is the generated sentence."
    result =get_chrf_precision_recall_fscore_support(reference_sentence,generated_sentence,[1,2])
    print(result)

