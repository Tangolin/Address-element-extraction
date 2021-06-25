### Address element extraction
---

This is a code made to complete the task on [Shopee Code League 2021.](https://www.kaggle.com/c/scl-2021-ds)

#### Files

1. train.csv - this is the original train data as provided by shopee.
2. new_train.pkl - pickled file of the pandas table after data cleaning. Feel free to check it out to see how it looks
3. aem.py - this is the main file to build and train the model.
4. train_model - this is the saved file of the trained model.

On my own training, it has achieve pretty good primary results in this NER task. That is it can recognise most of the entities in the sentence. However note that the output cannot be submitted directly, because it has yet to complete the part of the task which requires you to correct the spelling of the words to give the full form etc.
