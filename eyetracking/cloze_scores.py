import torch as tr
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from transformers import BertTokenizer, BertForMaskedLM, AutoTokenizer, AutoModelForMaskedLM


def get_cloze(model, target, tokenized_text, tokenizer, i_s, i_e, hidden_states=False, masked=False):
    """ Function to compute cloze scores and raw attention by applying a Transformer model
    Parameters
    ----------
    model: Transformer model such as "BertForMaskedLM.from_pretrained(modelpath)"
    target: single token for which cloze score is being calculated
    tokenized_text: entire sentence tokenized
    tokenizer: Tokenizer such as BertTokenizer.from_pretrained(modelpath, do_lower_case=False)
    i_s: first token to consider (usually ignoring [SEP] and [CLS] tokens)
    i_e: last token to consider
    hidden_states: BOOL whether to return raw attention matrix
    masked: BOOL whether to do masked language modelling

    Returns
    -------
    target: same as input
    prob: cloze value/word probability
    hidden_mat: raw attentions
    """
    x = tokenized_text
    ix = tokenizer.convert_tokens_to_ids([target])
    assert len(ix) == 1

    # Mask a token that we will try to predict back with `BertForMaskedLM`
    if masked:
        masked_index = tokenized_text.index(target)
        x[masked_index] = '[MASK]'

    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(x)
    # Convert inputs to PyTorch tensors
    tokens_tensor = tr.tensor([indexed_tokens])

    # Predict all tokens
    output = model(tokens_tensor, output_attentions=True, output_hidden_states=hidden_states)

    if masked:
        x[masked_index] = target

    probs = F.softmax(output.logits, dim=2)
    prob = probs[0, masked_index, ix].detach().numpy()

    if hidden_states:
        _hidden = [hidd.detach().numpy() for hidd in output.hidden_states]
        hidden_mat = np.asarray(_hidden)[:, 0]
        hidden_mat = hidden_mat[:, i_s:i_e]
    else:
        hidden_mat = None

    return target, prob, hidden_mat


def proc(df, data, hidden_states=False, masked=False):
    """Function to set up and collect cloze scores (word probability scores) based on Transformer model
    Parameters
    ----------
    df: pd.DataFrame with one word per row
    data: data as initialized in data.py
    hidden_states: BOOL whether to return raw attention matrix
    masked: BOOL whether to do masked language modelling

    Returns
    -------
    df_out: updated input "df" with cloze scores
    hidden_all: raw attention values (if hidden_states==True)
    """

    # Load pre-trained model tokenizer (vocabulary)
    modelpath = data.modelpath

    if 'roberta' in modelpath:
        tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large') #TODO
        model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-large") #TODO
    else:
        tokenizer = BertTokenizer.from_pretrained(modelpath, do_lower_case=False)
        model = BertForMaskedLM.from_pretrained(modelpath)

    model.eval()

    df_texts = df.groupby(['text_id'])
    df_out = pd.DataFrame(columns=['text_id', 'word', 'cloze', 'pos'])

    ii = 0
    hidden_all = []
    for itext, (text_id, df_text) in tqdm(enumerate(df_texts), total=len(df_texts)):

        text_raw = ' '.join(df_text.word)
        text = f'[CLS] {text_raw.lower()} [SEP]'
        tokenized_text = tokenizer.tokenize(text)

        #ignoring [CLS] and [SEP] tokens for cloze scores
        i_s = 1
        i_e = -1
        if 'xe ' in text.lower():
            tokenized_text[tokenized_text.index('x')] = 'xe'
            del tokenized_text[tokenized_text.index('xe')+1]
        hidden_out = [[] for _ in range(len(tokenized_text[i_s:i_e]))]
        look_for_ent = True

        for itarget, target in enumerate(tokenized_text[i_s:i_e]):
            ii += 1
            df_out, look_for_ent = data.assign_pos(df_out, ii, look_for_ent, itarget, target, tokenized_text, text_id)

            target, prob, hidden = get_cloze(model, target, tokenized_text, tokenizer, i_s, i_e,
                                             masked=masked,
                                             hidden_states=hidden_states)

            df_out.loc[ii, 'text_id'] = text_id
            df_out.loc[ii, 'word'] = target
            df_out.loc[ii, 'cloze'] = prob
            df_out.loc[ii, 'condition'] = df_text.iloc[0].condition if 'condition' in df_text else None
            df_out.loc[ii, 'noun'] = df_text.iloc[0].noun if 'noun' in df_text else None
            df_out.loc[ii, 'noungender'] = df_text.iloc[0].noungender if 'noungender' in df_text else None
            df_out.loc[ii, 'pronoun'] = df_text.iloc[0].pronoun if 'pronoun' in df_text else None
            df_out.loc[ii, 'ent'] = df_text.iloc[0].ent if 'ent' in df_text else None
            df_out.loc[ii, 'coref'] = df_text.iloc[0].coref if 'coref' in df_text else None
            df_out.loc[ii, 'stereotypical'] = df_text.iloc[0].stereotypical if 'stereotypical' in df_text else None
            hidden_out[itarget] = np.array(hidden)

        hidden_all.append(hidden_out)

    return df_out, hidden_all