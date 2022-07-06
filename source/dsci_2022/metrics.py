from rouge import Rouge
import pandas as pd

rouge = Rouge()

def compute_rouge(references, hypotheses):
    scores = rouge.get_scores(hyps = hypotheses, refs = references)
    dfs = [
        pd.DataFrame(d).T.assign(example = i)
        for i, d in enumerate(scores)
    ]
    full = pd.concat(dfs).reset_index()\
    .rename(
        columns = dict(
            index = "metric", 
            r = "recall", 
            p = "precision", 
            f = "f-measure"
        )
    )
    return full[["example", "metric", "f-measure", "precision", "recall"]]