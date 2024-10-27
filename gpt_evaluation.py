def get_results_select(demo, prompt, reltoid, idtoprompt,  args):
    #idtoprompt = {k:v for k,v in idtoprompt_ori.items()}
    #print(prompt)
    #assert False
    while True:
        try:
            results, probs = demo.get_multiple_sample(prompt)
            break
        except:
            return "Not generate properly."

    return results
    # if True:
    #     #return int(select_dict[results[0].strip()]), math.exp(probs[0]["token_logprobs"][0])
    #     #choice = [select_dict[i] for i in select_dict.keys() if results[0].strip() in select_dict[i]]]
    #     choice = 0
    #     for key, value in idtoprompt.items():
    #         if results.strip().strip(".").lower() in key.lower():
    #             choice = value

    #     #assert False
    #     print(results)
    #     #print(probs)
    #     print("the choice is ",choice)
    #     #if int(choice) == 7:
    #     #    print(results)
    #     #    assert False
    #     #print(choice)
    #     #return int(choice), math.exp(probs[0]["token_logprobs"][0]), probs[0], False
    #     return choice, 0.5, probs, False, results
    # else:
    #     print(prompt)
    #     print(results[0].strip())
    #     print(probs[0]["token_logprobs"][0])
    #     assert False

# def compute_f1(preds, labels):
#     n_gold = n_pred = n_correct = 0
#     for pred, label in zip(preds, labels):
#         if pred != 0:
#             n_pred += 1
#         if label != 0:
#             n_gold += 1
#         if (pred != 0) and (label != 0) and (pred == label):
#             n_correct += 1
#     if n_correct == 0:
#         return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
#     else:
#         prec = n_correct * 1.0 / n_pred
#         recall = n_correct * 1.0 /n_gold
#         if prec + recall > 0:
#             f1 = 2.0 * prec * recall / (prec + recall)
#         else:
#             f1 =0.0
#         return {"precision": prec, "recall": recall, "f1": f1}

def compute_f1(n_pred, n_gold, n_correct):
    
    if n_correct == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    else:
        prec = n_correct * 1.0 / n_pred
        recall = n_correct * 1.0 /n_gold
        if prec + recall > 0:
            f1 = 2.0 * prec * recall / (prec + recall)
        else:
            f1 =0.0
        return {"precision": prec, "recall": recall, "f1": f1}