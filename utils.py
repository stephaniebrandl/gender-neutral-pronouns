import numpy as np
from tqdm import tqdm
import spacy
import seaborn as sns

nlp = spacy.load("en_core_web_sm")
sns.color_palette("rocket", as_cmap=True)


class KeyDict(dict):
    def __missing__(self, key):
        return key


def extract_attentions(df, flow, group):
    if isinstance(group, tuple):
        group = list(group)
        
    link = {}
    perplexity = {}
    conds = []
    nlayers = flow[0].shape[1]

    for cond, df_cond in tqdm(df.groupby(group), total=len(df.groupby(group))):

        conds.append(cond)
        link[cond] = np.zeros([int(df_cond.text_id.nunique()), nlayers])
        perplexity[cond] = np.zeros(int(df_cond.text_id.nunique()))
        for itext_id, (text_id, df_textid) in tqdm(enumerate(df_cond.groupby('text_id')),
                                                   total=len(df_cond.groupby('text_id'))):

            df_tmp = df_textid.reset_index()
            ind_ent = df_tmp.query('pos=="ent"').index.to_list()
            ind_pron = df_tmp.query('pos=="pron"').index.to_list()

            if len(ind_pron) != 1:
                if 'they' in df_tmp.word.tolist():
                    ind_pron = df_tmp.word.tolist().index('they')
                elif 'xe' in df_tmp.word.tolist():
                    ind_pron = df_tmp.word.tolist().index('xe')
                elif 'han' in df_tmp.word.values[ind_pron] and 'de' in df_tmp.word.values[ind_pron]:
                    ind_pron = df_tmp.word.tolist().index('han')
                elif df_tmp.text_id.values[0] in [59, 79, 87, 147, 163, 230, 251, 263] and 'de' in df_tmp.word.tolist():
                    ind_pron = ind_pron[0]
                else:
                    raise ValueError('there should only be one index for each pronoun')

            flow_mat = np.array(flow[text_id])

            if len(ind_ent) == 1:
                link_mat = np.mean(flow_mat[:, :, ind_pron, ind_ent], axis=0)
            else:
                #max-pooling
                link_mat = np.mean(np.max(flow_mat[:, :, ind_pron, ind_ent], axis=2), axis=0)[:, None]
            link[cond][itext_id] = np.max(link_mat, axis=1)

            vals = df_tmp.cloze.values
            perplexity[cond][itext_id] += np.prod(vals ** (-1 / vals.shape[0]))

    return link, perplexity


def assign_stereotypical(df):
    assert(len(df==384))
    df['stereotypical'] = 0
    # those are the first indices of the stereotypical sentences of a block
    idx_stereotypical = [8, 24, 64, 208, 216, 248, 256, 280, 288, 304, 320, 376]
    for id in idx_stereotypical:
        df.loc[id:id + 7, 'stereotypical'] = 1

    #check whether the correct sentences have been selected

    stim_stereo = df.query('stereotypical==1')['Stimuli'].values
    noun_stereo = [stim_stereo[id].split(' ')[0] for id in range(stim_stereo.shape[0])]

    stim_nonstereo = df.query('stereotypical==0')['Stimuli'].values
    noun_nonstereo = [stim_nonstereo[id].split(' ')[0] for id in range(stim_nonstereo.shape[0])]

    overlap = [noun for noun in set(noun_stereo) if noun in set(noun_nonstereo)]
    assert (len(overlap) == 0)

    hardcoded_noun_stereo =[
        'Barnmorskan',
        'Byggnadsarbetaren',
        'Förskolläraren',
        'Målaren',
        'Officeraren',
        'Piloten',
        'Pizzabagaren',
        'Sekreteraren',
        'Sjuksköterskan',
        'Skönhetsterapeuten',
        'Snickaren',
        'Vårdbiträdet'
    ]

    missing = [noun for noun in hardcoded_noun_stereo if noun not in set(noun_stereo)]
    assert (len(missing) == 0)
    missing = [noun for noun in set(noun_stereo) if noun not in hardcoded_noun_stereo]
    assert (len(missing) == 0)

    idx_lexical = df.query("stereotypical==0 and noungender==-1").index
    for id in idx_lexical:
        df.loc[id, 'stereotypical'] = -1

    return df