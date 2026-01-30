# EricSearch()

## Initialization Arguments:

1. data_name (string) (optional). To load an already trained dataset, provide a Hugging Face ID as a string or a path to directory that contains a EricSearch() dataset. 

2. model_name (string) (optional): A Hugging Face ID or a path to a local embedding model. If data_name is provided, its value will be used unless it is overwritten with this parameter. The default is "sentence-transformers/all-MiniLM-L6-v2"

3. eric_ranker (EricRanker) (optional): An EricRanker object for information ranking.  

```python
from ericsearch import EricSearch, EricRanker

# For training: Initialize an empty dataset. 
eric_search_0 = EricSearch(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load an existing dataset, stored either locally or on Hugging Face's Hub
# The value provided to "embedding_name" when the dataset was initialized will be loaded. 
eric_search_1 = EricSearch(data_name="EricFillion/ericsearch-hello-world")

# Include a custom cross encoder. This model is not used for training, so it does not have to match the one that was initialized when the original dataset was trained. 
eric_ranker = EricRanker(model_name="cross-encoder/ms-marco-MiniLM-L6-v2")

eric_search_2 = EricSearch(data_name="EricFillion/ericsearch-hello-world", eric_ranker=eric_ranker)

```
## Call

1. text (string): Search query.

2. SearchCallArgs(): args object.

Where SearchCallArgs takes the following params:

1. limit (int) (1): Number of results

2. leaf_count (int) (32): The number of leaves that to search. Increase it to search for more relevant documents, at the cost of a slower search time. 

3. ranker_count (int) (4): The number of documents that are sent to EricRanker() for information extraction. Increasing this greatly decreases speed but improves accuracy.

4. bs (int) (32): Batch size for EricRanker's call. 


```python
from ericsearch import EricSearch, SearchCallArgs, RankerCallArgs

eric_search = EricSearch(data_name="EricFillion/ericsearch-hello-world")

args = SearchCallArgs(limit=2, leaf_count=32, ranker_count=4, bs=32)

result = eric_search("42", args=args)

print(result[0].text)  # str
print(result[0].best_sentence)  # str
print(result[0].metadata)  # dict 

```

The result is a list containing HDResult object, where the number returned is determined by limit. Each HDResult contains the following parameters: 

1. text: Typically a pargraph for the most relevant text.

2. best_sentence: The sentence with the highest score for the text parameter.

3. metadata: Any metadata provides during training that relates to the text parameter. 


## Train

### JSONL Format

The training data is provided as JSONL files with the following fields: 

1. text: A string that contains the text data for the case. 

2. metadata (optional): A dictionary that contains metadata for the specific case. Each metadata dictionary unique keys. 


```jsonl
{"text":  "TEXT 0 ", "metadata": {"id":  0, "other_data": "text"}}
{"text":  "TEXT 1 ", "metadata": {}}
{"text":  "TEXT 1 "}
```

### Code

EricSearch's .train() method takes the following parameters: 

1. train_path (str): A path to a directory that  contains JSONL files formatted properly.

```python
import os
import json

from ericsearch import EricSearch, SearchTrainArgs

train_dir = 'data/train'
out_dir = 'data/eric_search'

os.makedirs(train_dir, exist_ok=True)

eric_search = EricSearch()

train_data = []

for i in range(0, 100):
    train_data.append({"text": f"This is a sample train case {i}", "metadata": {"number": i}})


with open(f"{train_dir}/train.jsonl", "w", encoding="utf-8") as f:
    for train_case in train_data:
        f.write(json.dumps(train_case) + "\n")

args = SearchTrainArgs(leaf_size=4, out_dir=out_dir)

eric_search.train(train_path=train_dir, args=args)

d = eric_search("42")

print(d[0].text) 
```

## Push to Hugging Face's Hub

Use EricSearch's push method to push to Hugging Face's Hub. It has the following parameters: 

1. repo_id (str) (required): The Hugging Face ID

2. private (bool) (True): Repository's visibility. When True a private repository is made, when False it's public. 

3. bs (int) (4): Number of files that are pushed per commit. 

4. branch (str) (main): What branch to push to 

5. overwrite (bool) (False): Overwrite files in the Hugging Face repo. 


```python
from ericsearch import EricSearch

out_dir = 'data/eric_search'

# load from the out_dir from training 
eric_search = EricSearch(data_name=out_dir)

eric_search.push(repo_id={REPO ID GOES HERE}, private=True, bs=4)
```





