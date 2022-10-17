from rouge import Rouge
import pandas as pd

rouge_obj = Rouge()

def compute_rouge(references, hypotheses):
#     print(references)
#     print(hypotheses)
    scores = rouge_obj.get_scores(
        hyps = hypotheses, 
        refs = references, 
        ignore_empty = True,
        avg = True
    )
    print("scores", scores)
    return scores
    
# def compute_rouge(references, hypotheses):
#     print(references)
#     print(hypotheses)
#     empty = {
#         "rouge-1":{
#             "f":0,
#             "p":0,
#             "r":0,
#         },
#         "rouge-2":{
#             "f":0,
#             "p":0,
#             "r":0,
#         },
#         "rouge-l":{
#             "f":0,
#             "p":0,
#             "r":0,
#         },
#     }
#     scores = [
#         rouge_obj.get_scores(hyps = h, refs = r)
#         if h != "" and r != ""
#         else empty
#         for h, r in zip(hypotheses, references)
#     ]
#     dfs = [
#         pd.DataFrame(d).T.assign(example = i)
#         for i, d in enumerate(scores)
#     ]
#     full = pd.concat(dfs).reset_index()\
#     .rename(
#         columns = dict(
#             index = "metric", 
#             r = "recall", 
#             p = "precision", 
#             f = "f-measure"
#         )
#     )
#     return full[["example", "metric", "f-measure", "precision", "recall"]]