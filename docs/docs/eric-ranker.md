# EricRanker()

EricRanker() finds relevant paragraphs from provided texts.

## Initialization Arguments:

model_name (str) ("cross-encoder/ms-marco-MiniLM-L-6-v2"):  Either a Hugging Face ID or a path to a local directory where a cross encoder model is saved.

```python
from ericsearch import EricRanker, RankerCallArgs, EricDocument

eric_ranker = EricRanker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
```

## call()

Arguments: 

1. text (str) (required): A Search query 

2. docs (List[EricDocument]) (required): A list of EricDocument for the documents that will be searched. 

3. args (RankerCallArgs) (RankerCallArgs()): Settings

```python

from ericsearch import EricRanker, RankerCallArgs, EricDocument

eric_ranker = EricRanker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

text = "Hello world"

docs = [
    EricDocument(
    text="Eric Transformer is a Python package that supports fine-tuning and pretraining LLMs",
    score=1.0,
    metadata={"sample": "metadata"}),
    
    EricDocument(text="Kingston was once the capital of Canada. ",
                  score=0.5,  
                  metadata={"sample": "metadata"})
]


ranker_args = RankerCallArgs(bs=32, limit=1)
result = eric_ranker(text=text, docs=docs, args=ranker_args)

print(result[0].text)
print(result[0].best_sentence)
print(result[0].score)
print(result[0].metadata)


```

## EricDocument()

Arguments: 

1. text (str) (required): A string that contains the document 

2. score (float) (0.0): An overall score for the full document. This score is used to determine 60% of the final score for each paragraph.  Provide a float between 0 to 1 for each document, or 0 for all documents to disable this feature. 

3. metadata (dict) (optional): A dictionary that contains metadata for the document. 


## RankerCallArgs()

Arguments: 

1. bs (int) (32): Number of cases that are computed per batch with the cross encoder model. Increasing this value could speed up inference, but increases memory usage. 

2. limit (int) (32): Number of results to return. 






