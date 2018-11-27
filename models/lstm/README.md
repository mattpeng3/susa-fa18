# LSTM Model
An improved deep neural model with bidirectional LSTM cells and one-dimensional convolutional layers.

Originally written as a deep learning system to identify and classify toxic online comments. Heavily based on [work done by Gautam Mittal](https://github.com/gmittal/toxic-comments).

Data is from a [Kaggle competition](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) hosted by [Google Jigsaw](https://jigsaw.google.com/)'s [Conversation AI](https://conversationai.github.io/) team.

### Installation
Install dependencies with ```pip```.
```
pip install -r requirements.txt
```

### Usage
To train a new model, simply run the following. A pre-trained model checkpoint is included with this repository.

```
python train.py
```

To try the model out, run:
```
python evaluate.py -c "You're dumb"
```
To generate a ```submission.csv``` prediction file based on the [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) ```test.csv``` data, simply run:
```
python evaluate.py
```

### Model
Below is the network architecture used for classification. It was trained on [data from Google](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) available on Kaggle. **Achieves approximately 98% test accuracy** on both validation data and the public Kaggle leaderboard test data.

<p align="center">
<img width="300" src="save/model.png"/>
</p>


### License
The MIT License (MIT)

Copyright (c) 2018 Gautam Mittal

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
