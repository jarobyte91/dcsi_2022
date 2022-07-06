import itertools as it

def show(x, n = 5):
    print(x.shape)
    return x.head(n)

def extract_spans(tokens, cutoff = 0.5):
    assert all([len(x) == 2 for x in tokens])
    words = [w for w, s in tokens]
    scores = [s for w, s in tokens]
    keys = list(it.accumulate([s < cutoff for s in scores], initial = 0))
    grouped = [
        list(g) for k, g in it.groupby(
            zip(words, scores, keys), 
            lambda x: x[2]
        )
    ]    
    return [[(w, s) for w, s, k in l] for l in grouped if len(l) > 1]

def produce_summary(spans, sentence_separator = " ", summary_separator = "\n"):
    text = [sentence_separator.join([w for w, s in l]) for l in spans]
    return summary_separator.join(text)
    