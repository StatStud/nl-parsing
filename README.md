# How to run Entity Linking Algorithm

## First instantiate the class with desired variables (see code for documentation
```python
linker = entity_linker(embeddings_data="nodes/ddb_embeddings_synonym.pickle",
                 datafile = "nodes/ddb_nodes_header.csv")
```
                 
## Then choose a method of parsing

### Manually

```python
text = 
'''
A 35-year-old man comes to the physician because of itchy, 
watery eyes for the past week. He has also been sneezing multiple times a 
day during this period. He had a similar episode 1 year ago around springtime. 
He has iron deficiency anemia and ankylosing spondylitis. Current medications 
include ferrous sulfate, artificial tear drops, and indomethacin. He works as 
an elementary school teacher. His vital signs are within normal limits. Visual 
acuity is 20/20 without correction. Physical examination shows bilateral conjunctival 
injection with watery discharge. The pupils are 3 mm, equal, and reactive to light. 
Examination of the anterior chamber of the eye is unremarkable. Which of the following 
is the most appropriate treatment?
'''


linker.predict(text) 
```

The above returns list of linked entities for this entire context

### From an external script (.jsonl only, for now)

```python
linker.save_output(input_data = "data/medqa.jsonl", output_data = "test.jsonl")
```

The above snippet will create a copy of medqa.jsonl with the included parsed fields,
and even compute accuracy if ground truth entities are present.

# Tips on installing requirements

Both allennlp==2.1.0 allennlp-models==2.1.0 do not like to play with spacy==3.4.3.
Ignore this warning sign and continue anyway. It's best to first pip install btoh allennlp==2.1.0 allennlp-models==2.1.0, 
and then later, on another line, install spacy==3.4.3. That's simply how we're able to make this work.
                 
                 
