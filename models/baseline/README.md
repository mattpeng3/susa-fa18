### Baseline Toxicity Model
An intuitive baseline model is to correlate toxicity with the frequency of obscene words found in comment threads on social networks (manual feature detector).

```python
def predict(comment):
    """
    >>> predict('you are amazing!')
    0.0
    >>> predict('**** you ************')
    1.0    
    """
    words = len([word for word in comment.split() if word not in STOP_WORDS])
    obscene = len([word for word in comment.split() if word in BAD_WORDS])
    return 0 if words == 0 else obscene/words
```

