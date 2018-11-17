import pandas as pd

comments = pd.read_csv('reddit-analyze.csv')
sr_info = pd.read_csv('subreddit-info.csv')

print(comments.groupby('subreddit').mean().nlargest(10, 'toxic'))
    
