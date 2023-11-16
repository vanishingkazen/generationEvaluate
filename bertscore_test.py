'''
:param reference_sentences: 是ground truth
:type reference_sentences: list,里面放多个str
:param generated_sentences: 模型生成的结果
:type hypothesis: list,里面放多个str

:return results 返回 precision,recall,f1,hashcode
'''

'''{
'precision': [0.6349790096282959, 0.8026740550994873],
'recall': [0.6267067193984985, 0.8684506416320801],
'f1': [0.6308157444000244, 0.8342678546905518],
'hashcode': 'distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.23.1)'
}
'''



import evaluate

def getBertscore(reference_sentences,generated_sentences):
    bertscore = evaluate.load("./metrics/bertscore")
    results = bertscore.compute(predictions=reference_sentences, references=generated_sentences, lang="zh", model_type = "bert-base-chinese")
    return results


if __name__ == '__main__':
    generated_sentences = ['This is a cat', 'This is a cat']
    reference_sentences = ["that is a cat", 'dog']
    result = getBertscore(reference_sentences,generated_sentences)
    print(result)




