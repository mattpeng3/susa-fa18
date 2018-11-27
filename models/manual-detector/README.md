### Baseline Toxicity Model
An intuitive baseline model would be to correlate toxicity with a weighted average of the occurrences of obscene words found in the comment threads on social networks.

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

