### Tag Recommendation System (supervised learning)
hackerearth deep learning #4 challenge

#### Problem Statement
HackerEarth wants to improve its customer experience by suggesting tags for any idea submitted by a participant for a given hackathon. Currently, tags can only be manually added by a participant. HackerEarth wants to automate this process with the help of machine learning. To help the machine learning community grow and enhance its skills by working on real-world problems, HackerEarth challenges all the machine learning developers to build a model that can predict or generate tags relevant to the idea/ article submitted by a participant.

You are provided with approximately 1 million technology-related articles mapped to relevant tags. You need to build a model that can generate relevant tags from the given set of articles.

### Download data
Download these two data set, unzip and place that in the folder in data:

```

1. Tag data - from given torrent file (in the data folder)
2. Fastai word2vec embeddings - https://www.kaggle.com/yekenot/fasttext-crawl-300d-2m
```

#### Data Description

The dataset consists of ‘train.csv ’, ‘test.csv’ and ‘sample_submission.csv’. Description of the columns in the dataset is given below:


Variable | Description 
--- | --- 
id | Unique id for each article
title | Title of the article
article | Description of the article (raw format)
tags | Tags associated with the respective article. If multiple tags are associated with an article then they are seperated by pipe.

#### Submission
The submission file submitted by the candidate for evaluation has to be in the given format. The submission file is in .csv format. Check sample_submission for details. Remember, incase of multiple tags for a given article, they are seperated by '|'. 

```
id,tags
HE-efbc27d,java|freemarker
HE-d1fd267,phpunit|pear|osx-mountain-lion
HE-ffd4152,javascript|jquery|ajax|onclick
HE-d3ab268,forms|select|dojo
HE-ed2fa45,php|mysql|login|locking|ip-address
```

Leaderboard score = (1/n)*sum(Fi_score) (i.e mean F1 score of all articles)