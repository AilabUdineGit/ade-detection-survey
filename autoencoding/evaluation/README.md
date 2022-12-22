## How to run

`python evaluate.py -P sample_file.tsv -NB`

`-B` for binary classification, `-N` for entity extraction

## `evaluate.py`

Expected input: a tab separated file (with header)
containing at least the "General" fields and the ones
needed to evaluate the predictions for one or more of
the following tasks: "Entity Extraction", "Binary Classification".

**Columns**

- **General**
  - `doc_id`: (optional) unique identifier for the document
  - `text`: full text of the document on which predictions are made


- **Entity Extraction**
  - `gold_str`: list of real entities in the text as strings (optional, for sanity check)
  - `pred_str`: list of predicted entities as strings (optional, for sanity check)
  - `gold_int`: list of real entities in the text, given as character indexes (start included, end excluded) + entity type (string)
  - `pred_int`: list of predicted entities, given as character indexes (start included, end excluded) + entity type (string)


       gold_int and pred_int must be serializations of valid python lists
       examples:
       [(0,5,"AE"),(10,15,"Drug")]
       [[0,5,"AE"],[10,15,"Drug"]]

- **Binary Classification**
  - `gold_class`: int, actual class label
  - `pred_class`: int, predicted class label
  
## `sample_file.tsv`

Example of input file contining real labels and predictions for both tasks.

## `sample_file_evaluation_report.txt`

Output file with all main metrics.

## `sample_file_all_ner_metrics.txt`

Output file with additional metrics for the entity extraction task.