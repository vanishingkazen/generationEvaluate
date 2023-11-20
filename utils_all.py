from rouge import Rouge
from nltk.translate.chrf_score import chrf_precision_recall_fscore_support
from evaluate import load
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu
from bleurt import score


def getRougeScore(reference_sentences, generated_sentences,
                  rouge_n='l',
                  lang='zh'):
    """
    基于word级别的召回率评估，使用召回率直接作为分数
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


def getChrfScore(reference_sentences, generated_sentences,
                 n_gram=3,
                 beta=2):
    """
    基于字符级别的召回率和精准率，beta控制的是召回率和精准率对最后分数比重的影响，现已根据论文设置n_gram=3，beta=2 为最优,

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
        if len(reference_sentence) < n_gram or len(generated_sentence) < n_gram:
            raise Exception('length of reference_sentence or generated_sentences < n_gram')
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


def dealData(references, candidates, language):    # 分词
    '''
    分词子函数
    接收references, candidates两个字符串列表并将其分词
    对于每个字符串列表处理：['This is a cat', 'this is a test'] -> [['This', 'is', 'a', 'cat'], ['this', 'is', 'a', 'cat']]
    :param references: 参考译文，字符串列表
    :param candidates: 候选译文，字符串列表
    :param language: 语言，可选'en', 'zh'。已弃用，改为通过判断字符串第一个字符是否为英文来判断语言。
    :return: 分词后的references和candidates
    '''
    references = references[:]
    candidates = candidates[:]    # 防止不同指标计算时对原始数据作出修改
    for i in range(len(references)):
        if references[i][0].encode('utf-8').isalpha():    # 区分中英文。关键字参数language失效。
            language = 'en'
        elif not references[i][0].encode('utf-8').isalpha():
            language = 'zh'
        if language == 'en':  # 英文直接分词
            references[i] = references[i].split(' ')
            candidates[i] = candidates[i].split(' ')
        elif language == 'zh':
            references[i] = [ch for ch in references[i]]
            candidates[i] = [ch for ch in candidates[i]]
        else:
            print('Only English or Chinese supported.')
            exit()
    return references, candidates


def getBleu(references: list, candidates: list, weights=(1,), language='en'):
    '''
    计算Bleu分数
    :param references: 参考译文，字符串列表
    :param candidates: 候选译文，字符串列表
    :param weights: n-gram所占权重。如(0.5, 0.5)表示1-gram和2-gram的w均为0.5，(1,)表示1-gram的w为1，（0, 1)表示2-gram的w为1。
    :param language: 语言，可选'en'，'zh'。已弃用，在dealData中判定语言。
    :return: 两个句子的Bleu分数

    :demo
        references = ['This is a easy test', 'Today is a good day', 'Time is up']
        candidates = ['This is a test', 'Today is a bad day', 'Time is up']
        print('Bleu:', getBleu(references, candidates))    # 默认weights=(1,)
    '''
    references, candidates = dealData(references, candidates, language)    # 调用deal_data分词
    # 计算BLEU分数
    score_list = []
    for i in range(len(references)):
        score = sentence_bleu([references[i]], candidates[i], weights=weights)
        score_list.append(score)
    return score_list


def getMeteor(references: list, candidates: list, language='en', alpha=0.9, beta=3.0, gamma=0.5):    # gamma=0可使两个一样的句子得分为1
    '''
    计算meteor分数
    :param references: 参考译文，字符串列表
    :param candidates: 候选译文，字符串列表
    :param language: 语言，可选'en'，'zh'。已弃用，在dealData中判定语言。
    :param alpha: 参数
    :param beta: 参数
    :param gamma: 参数
    :return: 两个句子间的Bleu分数

    :demo
        references = ['This is a easy test', 'Today is a good day', 'Time is up']
        candidates = ['This is a test', 'Today is a bad day', 'Time is up']
        print('Meteor gamma=0.5:', getMeteor(references, candidates))    # 对于两个一样的句子，默认情况下gamma!=0，Meteor得分接近但不为1
        print('Meteor gamma=0:', getMeteor(references, candidates), gamma=0)    # gamma=0

    :setting
        需要安装一些东西，用下面代码就行
        # import nltk
        # nltk.download('wordnet')


    '''
    references, candidates = dealData(references, candidates, language)    # 调用deal_data分词
    # 计算METEOR分数
    score_list = []
    for i in range(len(references)):
        score = meteor_score([references[i]], candidates[i], alpha=alpha, beta=beta, gamma=gamma)
        score_list.append(score)
    return score_list


def getBleurt(references, candidates, checkpoint='./BLEURT-20'):
    ''''
    tip: 使用此函数需在此函数文件的同级目录下:
        1、创建BLEURT-20文件夹，其中包括文件夹variables，文件bert_config.json，bleurt_config.json，saved_model.pb，sent_piece.model，sent_piece.vocab
        2、BLEURT文件下载地址https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip
    计算Bleurt分数
    :param reference: 参考译文，字符串
    :param candidate: 候选译文，字符串
    :param checkpoint: 检查点，默认为'./BLEURT-20'
    :return: 两个句子间的Bleurt分数

    :demo
        references = ['This is a easy test', 'Today is a good day', 'Time is up']
        candidates = ['This is a test', 'Today is a bad day', 'Time is up']
        print('Bleurt:', getBleurt(references, candidates))
    '''
    scorer = score.BleurtScorer(checkpoint)
    scores = scorer.score(references=references, candidates=candidates)
    # assert isinstance(scores, list) and len(scores) == 1
    return scores


def getScores(metrics:list,reference_sentences,generated_sentences):
    # ALLmetrics = ['rouge','chrf','bertscore',]
    result = {}
    for metric in metrics:
        if metric == 'rouge':
            result['rouge'] = getRougeScore(reference_sentences,generated_sentences)
        elif metric == 'chrf':
            result['chrf'] = getChrfScore(reference_sentences,generated_sentences)
        elif metric == 'bertscore':
            result['bertscore'] = getBertScore(reference_sentences,generated_sentences)
        elif metric == 'bleu':
            result['bleu'] = getBleu(reference_sentences, generated_sentences)
        elif metric == 'meteor':
            result['meteor'] = getMeteor(reference_sentences, generated_sentences)
        elif metric == 'bleurt':
            result['bleurt'] = getBleurt(reference_sentences, generated_sentences)
        else:
            raise Exception('no such metric like {}'.format(metric))
    return result


if __name__ == '__main__':
    pass
    # reference_sentence = ['你是小猫',"我是一个男孩"]
    # generated_sentence = ['你是小狗',"你是一个女孩"]
    # result =getChrfScore(reference_sentence,generated_sentence)
    # print(result)
    #
    # generated_sentences = ["this is a  test","hello world", "你好","2", "你好"]
    # references = ["this is the small test","hello world", "你好","1", "你 好"]
    # scoures = getBertScore(references,generated_sentences)
    # print(scoures)
    # #
    # generated_sentences = ["我是一个男孩", "你好","2", "你好"]
    # reference_sentences = ["你是一个男孩", "你好","1", "你 好"]
    # print(getRougeScore(reference_sentences, generated_sentences))
    try:
        generated_sentences = ["this is a  test","hello world"]
        references = ["this is the small test","hello world"]

        # metrics = ['rouge', 'chrf', 'bertscore', ]
        # metrics = ['bleurt', 'bleu', 'meteor']
        metrics = ['rouge', 'chrf', 'bertscore', 'bleurt', 'bleu', 'meteor']

    except Exception as e:
        print(e)
    print(getScores(metrics,references,generated_sentences))

    # import nltk
    # nltk.download('wordnet')
