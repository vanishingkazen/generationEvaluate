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
# https://blog.csdn.net/rainy_universe/article/details/128493300
from nltk.translate.chrf_score import chrf_precision_recall_fscore_support
# precision准确率 , recall 召回率, fscore, tp(true positive)
def get_chrf_precision_recall_fscore_support(reference_sentence,generated_sentence,n_gram_ls=[1]):
    result = []
    for n_gram in n_gram_ls:
        precision, recall, fscore, tp = chrf_precision_recall_fscore_support(
            reference_sentence, generated_sentence,n=n_gram,epsilon=0.,beta=2.0
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
    reference_sentence = '你 好'
    generated_sentence = '你 好'
    result =get_chrf_precision_recall_fscore_support(reference_sentence,generated_sentence,[2])
    print(result)

    # precision, recall, fscore, tp = chrf_precision_recall_fscore_support(
    #     'the cat is on the mat. my name is kk.', 'the the the the the the the', n=1, beta=3.0
    # )
    # print('the cat is on the mat. my name is kk.'.split(), 'the the the the the the the'.split())
    # print(precision, recall, fscore, tp)

