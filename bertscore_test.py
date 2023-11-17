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


from evaluate import load

def getBertscore(reference_sentences,generated_sentences):
    bertscore = load("./metrics/bertscore")
    # predictions = generated_sentences
    # references = reference_sentences
    results = bertscore.compute(predictions=generated_sentences, references=reference_sentences,lang="zh", model_type="bert-base-chinese")
    return results


if __name__ == '__main__':
    generated_sentences = ["this is a  test","hello world", "你好","2", "你好"]
    references = ["this is the small test","hello world", "你好","1", "你 好"]
    results = getBertscore(references,generated_sentences)
    print(results)





