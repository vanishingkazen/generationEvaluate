from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu
from bleurt import score
from rouge import Rouge
from nltk.translate.chrf_score import chrf_precision_recall_fscore_support
from evaluate import load


def getRougeScore(reference_sentence, generated_sentence,
                  rouge_n='l',
                  lang='zh'):
    """

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
    if reference_sentence == '' or generated_sentence == '':
        raise Exception("the sentence is empty")

    if lang == 'en':
        pass
    elif lang == 'zh':
        reference_sentence = " ".join(reference_sentence)
        generated_sentence = " ".join(generated_sentence)
    else:
        raise Exception("para:lang type error")

    rouge = Rouge()
    rouge_score = rouge.get_scores(generated_sentence, reference_sentence)
    return rouge_score[0]["rouge-{}".format(rouge_n)]['f']


def getChrfScore(reference_sentence, generated_sentence, n_gram=3, beta=2):
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
    if reference_sentence == '' or generated_sentence == '':
        raise Exception("the sentence is empty")
    precision, recall, fscore, tp = chrf_precision_recall_fscore_support(
        reference_sentence, generated_sentence, n=n_gram, epsilon=0., beta=beta
    )
    return fscore


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


def deal_data(reference, candidate, language):    # 分词
    '''
    子函数，将字符串分词。如'This is a cat' -> ['This', 'is', 'a', 'cat']
    :param reference: 字符串
    :param candidate: 字符串
    :param language: 语言，可选'English', 'Chinese'
    :return: 分词后的reference和candidate
    '''
    # 生成reference与candidate
    if language == 'English':  # 英文直接分词
        reference = [reference.split()]
        candidate = candidate.split()
    elif language == 'Chinese':  # 中文通过jieba库分词
        import jieba
        reference = [' '.join(jieba.cut(reference)).split()]
        candidate = ' '.join(jieba.cut(candidate)).split()
    else:
        print('Only English or Chinese supported.')
        exit()
    return reference, candidate


def get_pr(reference: str, candidate: str):
    '''
    计算准确率与召回率
    :param reference: 参考译文
    :param candidate: 候选译文
    :return: p：准确率，r：召回率
    '''
    m, r, c = 0., float(len(reference)), float(len(candidate))    # 匹配数，ref词数，can词数
    for word in candidate:
        if word in reference:
            m += 1
    p = float(m) / c
    r = m / r
    return p, r


def get_bleu(reference: str, candidate: str, weights=(0.25, 0.25, 0.25, 0.25), language='English'):
    '''
    获取Bleu分数
    :param reference: 参考译文，字符串
    :param candidate: 候选译文，字符串
    :param weights: n-gram所占权重。(0.5, 0.5)表示1-gram和2-gram的w均为0.5。
    :param language: 语言，可选'English'，'Chinese'
    :return: 两个句子的Bleu分数与准确率、召回率
    '''
    reference, candidate = deal_data(reference, candidate, language)    # 调用deal_data分词
    # 计算准确率、召回率
    p, r = get_pr(reference[0], candidate)
    # 计算BLEU分数
    score = sentence_bleu(reference, candidate, weights=weights)
    return score, p, r


def get_meteor(reference: str, candidate: str, language='English', alpha=0.9, beta=3.0, gamma=0.5):    # gamma=0可使两个一样的句子得分为1
    '''
    获取meteor分数
    :param reference: 参考译文，字符串
    :param candidate: 候选译文，字符串
    :param language: 语言，可选'English'，'Chinese'
    :param alpha: 超参
    :param beta: 超参
    :param gamma: 超参
    :return: 两个句子间的Bleu分数与准确率、召回率
    '''
    reference, candidate = deal_data(reference, candidate, language)    # 调用deal_data分词
    # 计算准确率，召回率
    p, r = get_pr(reference[0], candidate)
    # 计算METEOR分数
    score = meteor_score(reference, candidate, alpha=alpha, beta=beta, gamma=gamma)
    return score, p, r


def getBleurt(reference, candidate):
    '''
    计算Bleurt分数
    :param reference: 参考译文，字符串
    :param candidate: 候选译文，字符串
    :return: 两个句子间的Bleurt分数与准确率、召回率
    '''
    references, candidates = [reference], [candidate]
    scorer = score.BleurtScorer()
    scores = scorer.score(references=references, candidates=candidates)
    return scores


def automatic(reference, candidate, params:list):    # 输入列表
    '''
    入口函数
    :param reference: 参考译文，字符串
    :param candidate: 候选译文，字符串
    :param params: 要计算的标准，列表
    :return: 所要计算的标准的得分列表
    '''
    result = []
    # name = ['score', 'p', 'r']
    name = ['score']
    for param in params:
        if param == 'Bleu':
            score, precision, recall = get_bleu(reference, candidate)
            number = [score, precision, recall]
            dic_bleu = dict(zip(name, number))
            result.append({'Bleu': dic_bleu})
        elif param == 'Meteor':
            score, precision, recall = get_meteor(reference, candidate)
            number = [score, precision, recall]
            dic_bleu = dict(zip(name, number))
            result.append({'Meteor': dic_bleu})
        elif param == 'Bleurt':
            score = getBleurt(reference, candidate)
            result.append({'Bleurt:': score[0]})
        else:
            print('请输入正确的评估标准！')
            exit()
    return result


if __name__ == '__main__':
    reference = 'This is a small test without hard problems'
    candidate = 'This is a easy test and three are no hard quertions'
    # print(get_bleu(reference, candidate))
    # print(get_meteor(reference, candidate))
    res = automatic(reference, candidate, ['Bleu', 'Meteor', 'Bleurt'])
    for item in res:
        print(item)

