# How Conservative are Language Models? Adapting to the Introduction of Gender-Neutral Pronouns  

This repository contains code connected to the paper ["How Conservative are Language Models? Adapting to the Introduction of Gender-Neutral Pronouns."](https://aclanthology.org/2022.naacl-main.265.pdf) accepted at NAACL 2022. 

Please refer to the paper for further details.

## Perplexity scores and attention flow
So far, we provide code to compute perplexity scores and correlate them with pre-computed 
flow values. To carry out this analysis for Swedish, you first need to download the file 
"HEN_preprocessing_scripts/HENprepdata.txt" [here](https://su.figshare.com/articles/dataset/Open_data_Are_new_gender-neutral_pronouns_difficult_to_process_in_reading_The_case_of_hen_in_Swedish/13143158) 
and include the local data path into `config.yaml`. For Danish and English you find the translated version of that dataset here in the repo under `data/stimuli/{lan}_translated_{pron}.pkl`.  

In `config.yaml` you also need to set the language (currently: en, da, sv) and the corresponding pronoun (they/xe, de/høn, hen respectively) and then run `run_perplexity_flow.py`