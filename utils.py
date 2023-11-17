from rouge import Rouge
from nltk.translate.chrf_score import chrf_precision_recall_fscore_support
from evaluate import load


def getRougescore(reference_sentences,generated_sentences,rouge_n='l'):
    rouge = Rouge()
    rouge_score = rouge.get_scores(generated_sentences, reference_sentences)
    return rouge_score[0]["rouge-{}".format(rouge_n)]['f']

def get_chrf_precision_recall_fscore_support(reference_sentence,generated_sentence,n_gram=3,beta=2,lang = 'zh'):
    if(lang == 'en'):
        pass
    elif(lang == 'zh'):
        reference_sentence = [c for c in reference_sentence]
        generated_sentence = [c for c in generated_sentence]
    else:
        raise Exception("para:lang type error")
    precision, recall, fscore, tp = chrf_precision_recall_fscore_support(
            reference_sentence, generated_sentence,n=n_gram,epsilon=0.,beta=beta
    )
    return fscore

def getBertscore(reference_sentences,generated_sentences):
    bertscore = load("./metrics/bertscore")
    # predictions = generated_sentences
    # references = reference_sentences
    results = bertscore.compute(predictions=generated_sentences, references=reference_sentences,lang="zh", model_type="bert-base-chinese")
    return results


if __name__ == '__main__':
    # reference_sentence = '你是小猫'
    # generated_sentence = '你是小狗'
    # result =get_chrf_precision_recall_fscore_support(reference_sentence,generated_sentence)
    # print(result)

    generated_sentences = ["this is a  test","hello world", "你好","2", "你好"]
    references = ["this is the small test","hello world", "你好","1", "你 好"]
    results = getBertscore(references,generated_sentences)
    print(results)