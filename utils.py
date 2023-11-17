from rouge import Rouge
from nltk.translate.chrf_score import chrf_precision_recall_fscore_support
from evaluate import load


def getRougeScore(reference_sentences, generated_sentences,
                  rouge_n='l',
                  lang='zh'):
    """
    使用召回率直接作为分数
    :param reference_sentence: 传入一个str字符串
    :param generated_sentence: 传入一个str字符串
    :param rouge_n: 可选 '1','2','l',设置算法中的 n_gram=？ ,默认使用 rouge-l 的 lcs最长公共子序列的算法
    :param lang: 设置语言， 可选'en','zh'
    :return: 返回一个分数

    :demo
        generated_sentences = "我是一个男孩"
        reference_sentences = "你是一个男孩"
        print(getRougescore(reference_sentences, generated_sentences))
    """
    scores = []
    rouge = Rouge()

    for generated_sentence, reference_sentence in zip(generated_sentences, reference_sentences):
        if lang == 'en':
            pass
        elif lang == 'zh':
            reference_sentence = " ".join(reference_sentence)
            generated_sentence = " ".join(generated_sentence)
        else:
            raise Exception("para:lang type error")
        rouge_score = rouge.get_scores(generated_sentence, reference_sentence)
        scores.append(rouge_score[0]["rouge-{}".format(rouge_n)]['p'])
    return scores


def getChrfScore(reference_sentences, generated_sentences, n_gram=3, beta=2):
    """

    :param reference_sentence: 传入一个str字符串
    :param generated_sentence: 传入一个str字符串
    :param n_gram: 默认为3
    :param beta: 用于调节 recall和 precise 作用于分数的权重
    :return: 返回一个分数

    :demo
        reference_sentence = '你是小猫'
        generated_sentence = '你是小狗'
        result =get_chrf_precision_recall_fscore_support(reference_sentence,generated_sentence)
        print(result)
    """
    fscores = []
    for reference_sentence, generated_sentence in zip(reference_sentences, generated_sentences):
        precision, recall, fscore, tp = chrf_precision_recall_fscore_support(
            reference_sentence, generated_sentence, n=n_gram, epsilon=0., beta=beta
        )
        fscores.append(fscore)
    return fscores


def getBertScore(reference_sentences, generated_sentences):
    """
    tip:使用此函数需要在此函数文件的同级目录下 1.创建模型文件夹"bert-base-chinese",模型文件夹内有三个文件分别是:config.json,pytorch_model.bin,vocab.txt
                                        2.需要evaluate的源码中的metrics文件夹中的脚本文件，可以在git中下载 https://github.com/huggingface/evaluate

    :param reference_sentences: list,传入任意数量的字符串，注意数量要和 generated_sentences 中的数量保持一致
    :param generated_sentences: list,传入任意数量的字符串，注意数量要和 reference_sentences 中的数量保持一致
    :return: 返回分数列表，列表中分数元素的个数和 generated_sentences 中的元素数量相同

    demo：
    generated_sentences = ["this is a  test","hello world", "你好","2", "你好"]
    references = ["this is the small test","hello world", "你好","1", "你 好"]
    scoures = getBertscore(references,generated_sentences)
    print(scoures)
    """
    if len(reference_sentences) == 0 or len(generated_sentences) == 0:
        raise Exception("the sentences list is empty")
    bertscore = load("./metrics/bertscore")
    results = bertscore.compute(predictions=generated_sentences, references=reference_sentences, lang="zh",
                                model_type="bert-base-chinese")
    scores = results['f1']
    return scores


if __name__ == '__main__':
    # pass
    reference_sentence = ['你是小猫',"我是一个男孩"]
    generated_sentence = ['你是小狗',"你是一个女孩"]
    result =getChrfScore(reference_sentence,generated_sentence)
    print(result)

    generated_sentences = ["this is a  test","hello world", "你好","2", "你好"]
    references = ["this is the small test","hello world", "你好","1", "你 好"]
    scoures = getBertScore(references,generated_sentences)
    print(scoures)

    generated_sentences = ["我是一个男孩", "你好","2", "你好"]
    reference_sentences = ["你是一个男孩", "你好","1", "你 好"]
    print(getRougeScore(reference_sentences, generated_sentences))
