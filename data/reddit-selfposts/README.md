# Reddit Self-posts
An ImageNet-like text classification task based on reddit posts. This dataset is too large to upload to GitHub. [Download it directly](https://www.kaggle.com/mswarbrickjones/reddit-selfposts/) from Kaggle.

### Introduction
Welcome to the Reddit Self-Post Classification Task (RSPCT)!

The aim of this dataset was to create an interesting, large text classification problem with many classes, that does not suffer from label sparsity as most datasets of its type do. See the blog post for a more detailed write up, or the paper here. The aim is to classify self-posts into the subreddit into which they were posted. A great deal of effort has gone into selecting a ‘good’ set of subreddits to minimize overlap in content.

We recommend you look at the [blogpost write-up](https://www.evolution.ai/blog/page/5/an-imagenet-like-text-classification-task-based-on-reddit-posts/) for this dataset before continuing. There is also a rough draft of a paper [here](https://www.evolution.ai/blog_figures/reddit_dataset/rspct_preprint_v3.pdf) if you have more detailed questions.

### Data
The data consists of 1.013M self-posts, posted from 1013 subreddits (1000 examples per class). For each post we give the subreddit, the title and content of the self-post.

We have also given a manual annotation of about 3000 subreddits which went into the creation of this dataset, in subreddit_info.csv, this was the main criteria for selecting which subreddits went into this dataset. We include a top-level category and subcategory for each subreddit, and a reason for exclusion if this does not appear in the data.
