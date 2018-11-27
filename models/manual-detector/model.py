"""
Baseline manual feature detector.
Predicts toxicity based on frequency of obscene words.
Written by Gautam Mittal
"""

STOP_WORDS = set(open('./stopwords.txt', 'r').read().split('\n')[:-1])
BAD_WORDS = set(open('./badwords.txt', 'r').read().split('\n')[:-1])

def predict(comment):
    words = len([word for word in comment.split() if word not in STOP_WORDS])
    obscene = len([word for word in comment.split() if word in BAD_WORDS])
    return 0 if words == 0 else obscene/words
