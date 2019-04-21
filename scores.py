import numpy as np
import collections
import sys

test_raw = sys[1]
predict_raw = sys[2]

labels = ["slightly angry", "fairly angry", "extremely angry",\
         "slightly fearful", "fairly fearful", "extremely fearful",\
          "slightly joyful", "fairly joyful", "extremely joyful",\
          "slightly sad", "fairly sad", "extremely sad"]

mood = {'anger':'angry', 'fear':'fearful', 'joy':'joyful', 'sadness':'sad'}

def label(its, noun):
    if its >= 0 and its < 0.33:
        return "slightly " + mood[noun]
    elif its >= 0.33 and its < 0.67:
        return "fairly " + mood[noun]
    else:
        return "extremely " + mood[noun]

def f1_score(prediction, ground_truth):
    prediction_tokens =  prediction.split()
    ground_truth_tokens =  ground_truth.split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match(prediction, ground_truth):
    return prediction == ground_truth

mood_score={"slightly":"zero zero","fairly":"zero one","extremely":"one one"}
def f1_em(prediction, ground_truth):
    prediction_tokens =  prediction.split()
    ground_truth_tokens =  ground_truth.split()
    if len(prediction_tokens)!=2 or prediction_tokens[0] not in mood_score.keys():
        return 0
    
    pred_intensity=mood_score[prediction_tokens[0]]
    truth_intensity=mood_score[ground_truth_tokens[0]]
    return exact_match(prediction_tokens[1],ground_truth_tokens[1])*f1_score(pred_intensity,truth_intensity)

with open(test_raw) as testraw:
    tests = testraw.read().split('\n')[:]

with open(predict_raw) as prediction:
    predicts = prediction.read().split('\n')[:-1]

f1 = em = f1em = 0
for i in range(len(tests)):
    predict_line = predicts[i]
    predict_list = predict_line.split('\t')
    predict = predict_list.index(max(predict_list))
    
    test_line = tests[i]
    test_list = test_line.split('\t')
    
    predict_label = labels[predict]
    test_label = label(float(test_list[3]), test_list[2])
    
    f1 += f1_score(predict_label, test_label)
    em += exact_match(predict_label, test_label)
    f1em += f1_em(predict_label, test_label)

em = 100.0 * em / len(tests)
f1 = 100.0 * f1 / len(tests)
f1em = 100.0 * f1em / len(tests)

results = {"exact_match":em, "f1_score":f1, "f1_em":f1em}

with open('test_score.txt', 'w') as scores:
    for i in results:
        scores.write(i + ": " + str(results[i]) + '\n')

print(results)