import itertools as it
import re

def show(x, n = 5):
    print(x.shape)
    return x.head(n)

def extract_spans(tokens, threshold = 0.5):
    assert all([len(x) == 2 for x in tokens])
    words = [w for w, s in tokens]
    scores = [s for w, s in tokens]
    keys = list(it.accumulate([s < threshold for s in scores], initial = 0))
    grouped = [
        list(g) for k, g in it.groupby(
            zip(words, scores, keys), 
            lambda x: x[2]
        )
    ]    
    return [[(w, s) for w, s, k in l] for l in grouped if len(l) > 1]

# def produce_summary(spans, sentence_separator = " ", summary_separator = "\n"):
#     text = [sentence_separator.join([w for w, s in l]) for l in spans]
#     return summary_separator.join(text)

def score_spans(tokens, tokenizer, threshold = 0.5):
    spans = extract_spans(
        tokens, 
        threshold = threshold,
    )
    spans_scored = [
        (
            sum([s for w, s in l]) / len(l), 
            [w for w, s in l]
        ) 
        for l in spans
    ]
    return sorted(
        [(s, tokenizer.decode(l)) for s, l in spans_scored], 
        key = lambda x: x[0], 
        reverse = True
    )
    
def detokenize(
    tokens, 
    reference_scores, 
    predicted_scores, 
    tokenizer, 
    threshold = 0.5
):
    reference_spans = extract_spans(
        list(zip(tokens, reference_scores)), 
        threshold = threshold
    )
#     print([len(s) for s in reference_spans])
    hypothesis_spans = extract_spans(
        list(zip(tokens, predicted_scores)),
        threshold = threshold
    )
#     print([len(s) for s in hypothesis_spans])
    reference_summary = " ".join(
        [tokenizer.decode([t[0] for t in s]) for s in reference_spans]
    )
#     print(len(reference_summary))
#     print([len(s) for s in reference_summary])

#     print(reference_summary[:500])
    hypothesis_summary = " ".join(
        [tokenizer.decode([t[0] for t in s]) for s in hypothesis_spans]
    )
#     print(len(hypothesis_summary))
#     print(hypothesis_summary[:500])
    reference_summary = re.sub(r"\[.*?\]", "", reference_summary)
    hypothesis_summary = re.sub(r"\[.*?\]", "", hypothesis_summary)
    return reference_summary, hypothesis_summary

    