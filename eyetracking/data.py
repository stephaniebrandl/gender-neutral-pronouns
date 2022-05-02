import pandas as pd
import numpy as np
import pickle
from googletrans import Translator
from tqdm import tqdm
from os import path
from os import makedirs
from eyetracking.utils import read_yaml

config = read_yaml('config.yaml')


class SWEyetracking():
    def __init__(self, path=None):
        if path==None:
            path = config['data_path']['eyetracking']

        df = pd.read_csv(path, delimiter='\t')
        self.df = df.query("BadTrial==0")
        self.modelpath = 'af-ai-center/bert-base-swedish-uncased'
        self.dicts = self._dicts()
        self.keys = self.dicts.keys()
        self.name = 'sw_eyetracking'
        self.pronoun = 'hen'

    def _dicts(self):
        dicts = {}
        dicts['pronoun'] = {-1: 'hon/han', 1: 'hen'}

        return dicts

    def preprocess_df(self):
        df_fixation = self.df.groupby('Stimuli').mean()[
            ['PronounFix', 'NounFix', 'PronounFirstFix', 'NounFirstFix', 'NounRevFix', 'NounRevisits',
             'Condition', 'nounfemmasc', 'pron', 'noungender']].reset_index()
        # df_fixation = assign_stereotypical(df_fixation)
        df_fixation = df_fixation.rename(columns={'pron': 'pronoun'})

        df_words = df_fixation.assign(Stimuli=df_fixation['Stimuli'].str.split(' ')).explode('Stimuli').reset_index()
        df_words = df_words.rename(columns={'index': 'text_id', 'Stimuli': 'word', 'Condition': 'condition',
                                            'nounfemmasc': 'noun', 'pron': 'pronoun'})

        return df_fixation, df_words

    def assign_pos(self, df, ii, look_for_ent, itarget, target, tokenized_text, text_id):
        if look_for_ent:
            if (
                    itarget == 0 or
                    target.startswith(("-", "##", "år")) or
                    tokenized_text[itarget + 1].startswith("-") or
                    tokenized_text[itarget] == (("70"))) or \
                    (text_id in np.arange(272, 279) and target == 'läkaren'):
                df.loc[ii, 'pos'] = 'ent'
            else:
                look_for_ent = False
                df.loc[ii, 'pos'] = 'unk'
        elif target in ['han', 'hon', 'hen']:
            df.loc[ii, 'pos'] = 'pron'
        else:
            df.loc[ii, 'pos'] = 'unk'

        return df, look_for_ent


class DAEyetracking():
    def __init__(self, pronoun='høn'):
        self.pronoun = pronoun
        try:
            self.df_fixation = pd.read_pickle(f'./data/stimuli/da_translated_{self.pronoun}.pkl')
        except FileNotFoundError:
            print('file not found, start translating')
            self.df_fixation = pd.DataFrame()
            for ii in range(4):
                idx = np.arange(ii*100, np.min([(ii+1)*100, 384]))
                df = self._translate(idx)
                self.df_fixation = self.df_fixation.append(df.iloc[idx], ignore_index=True)
            self._save_df()

        self.modelpath = "Maltehb/danish-bert-botxo"
        self.dicts = self._dicts()
        self.keys = self.dicts.keys()

    def _dicts(self):
        dicts = {}
        dicts['pronoun'] = {-1: 'hun/han', 1: self.pronoun}

        return dicts

    def _translate(self, idx=None):
        translator = Translator(service_urls=[
            'translate.google.dk'
        ])
        sw = SWEyetracking()
        df_fixation_sw, _ = sw.preprocess_df()
        df_fixation = df_fixation_sw[['Condition', 'nounfemmasc', 'noungender', 'stereotypical', 'pronoun']].copy()
        if idx is None:
            idx = np.arange(0, 384)
        for irow, row in tqdm(df_fixation_sw.iloc[idx].iterrows(), total=len(df_fixation_sw.iloc[idx])):
            if self.pronoun in ['de', 'De']:
                translated = translator.translate(row['Stimuli'].replace('Hen', 'De').replace('hen', 'de'),
                                                  src='sv', dest='da')
            elif self.pronoun in ['Høn', 'høn']:
                translated = translator.translate(row['Stimuli'].replace('Hen', 'Høn').replace('hen', 'høn'),
                                                  src='sv', dest='da')
            df_fixation.loc[irow, 'Stimuli'] = translated.text.replace('.', '. ').replace('Stordenen', 'Læge')\
                .replace('stordenen', 'læge')
        return df_fixation

    def _save_df(self):
        out_dir = './data/stimuli'
        if not path.isdir(out_dir):
            makedirs(out_dir)
        filename = f'da_translated_{self.pronoun}.pkl'
        pickle.dump(self.df_fixation, open(path.join(out_dir, filename), "wb"))

    def preprocess_df(self):
        df_words = self.df_fixation.assign(
            Stimuli=self.df_fixation['Stimuli'].str.split(' ')).explode('Stimuli').reset_index()
        df_words = df_words.rename(columns={'index': 'text_id', 'Stimuli': 'word', 'Condition': 'condition',
                                            'nounfemmasc': 'noun'})
        return self.df_fixation, df_words

    def assign_pos(self, df, ii, look_for_ent, itarget, target, tokenized_text, text_id):
        if look_for_ent:
            if (
                    itarget == 0 or
                    target.startswith(('▁')) or
                    target.startswith(("-", "##", "år")) or #TODO
                    tokenized_text[itarget + 1].startswith("-") or
                    tokenized_text[itarget] == (("70"))) or \
                    (text_id in np.arange(312, 320) and target == 'pårørende'
                    ) or \
                    (text_id in np.arange(328, 336) and target == 'studerende'
                    ):
                df.loc[ii, 'pos'] = 'ent'
            else:
                look_for_ent = False
                df.loc[ii, 'pos'] = 'unk'
        elif target in ['han', 'hun', self.pronoun]:
            df.loc[ii, 'pos'] = 'pron'
        else:
            df.loc[ii, 'pos'] = 'unk'

        return df, look_for_ent


class ENEyetracking():
    def __init__(self, pronoun='they'):
        self.pronoun = pronoun
        try:
            self.df_fixation = pd.read_pickle(f'./data/stimuli/en_translated_{self.pronoun}.pkl')
        except FileNotFoundError:
            print('file not found, start translating')
            self.df_fixation = pd.DataFrame()
            for ii in range(4):
                idx = np.arange(ii*100, np.min([(ii+1)*100, 384]))
                df = self._translate(idx)
                self.df_fixation = self.df_fixation.append(df.iloc[idx], ignore_index=True)
            self._save_df()

        self.modelpath = 'bert-base-uncased'
        self.dicts = self._dicts()
        self.keys = self.dicts.keys()

    def _dicts(self):
        dicts = {}
        dicts['pronoun'] = {-1: 'she/he', 1: self.pronoun}

        return dicts

    def _translate(self, idx=None):
        translator = Translator(service_urls=[
            'translate.google.se'
        ])
        sw = SWEyetracking()
        df_fixation_sw, _ = sw.preprocess_df()
        df_fixation = df_fixation_sw[['Condition', 'nounfemmasc', 'noungender', 'stereotypical', 'pronoun']].copy()
        if idx is None:
            idx = np.arange(0, 384)
        for irow, row in tqdm(df_fixation_sw.iloc[idx].iterrows(), total=len(df_fixation_sw.iloc[idx])):
            if self.pronoun in ['they', 'They']:
                translated = translator.translate(row['Stimuli'].replace('Hen', 'They').replace('hen', 'they'), src='sv')
            elif self.pronoun in ['Xe', 'xe']:
                translated = translator.translate(row['Stimuli'].replace('Hen', 'Xe').replace('hen', 'xe'), src='sv')
            df_fixation.loc[irow, 'Stimuli'] = translated.text.replace('.', '. ').replace('They was', 'They were')\
                .replace('they was', 'they were').replace('They has', 'They have').replace('they has', 'they have')\
                .replace('They is', 'They are').replace('they is', 'they are')
        return self._post_correction(df_fixation, idx)

    def _post_correction(self, df_fixation, idx):
        for irow, row in tqdm(df_fixation.iloc[idx].iterrows(), total=len(df_fixation.iloc[idx])):
            df_fixation.loc[irow, 'Stimuli'] = row['Stimuli'].replace('his 80th birthday', '80th birthday') \
                .replace('felt almost everyone', 'knew almost everyone') \
                .replace('wanted to do his best', 'wanted to do best') \
                .replace('for his buddy', 'for a buddy').replace('about his relationship', 'about the relationship') \
                .replace('its 80th birthday', '80th birthday') \
                .replace('asking his boss', 'asking the boss')\
                .replace('mother embroidery', 'uncle').replace('mother embroider', 'uncle').replace('morbigrodern', 'uncle')\
                .replace('Morbruder', 'uncle').replace('Moster', 'aunt').replace('Mostern', 'aunt')\
                .replace('pizza bags', 'pizza maker').replace('pizza bager', 'pizza maker')\
                .replace('pizza bayer', 'pizza maker').replace('nursing asset', 'caregiver')\
                .replace('caring assistant', 'caregiver')
        return df_fixation

    def _save_df(self):
        out_dir = './data/stimuli'
        if not path.isdir(out_dir):
            makedirs(out_dir)
        filename = f'en_translated_{self.pronoun}.pkl'
        pickle.dump(self.df_fixation, open(path.join(out_dir, filename), "wb"))

    def preprocess_df(self):
        df_words = self.df_fixation.assign(
            Stimuli=self.df_fixation['Stimuli'].str.split(' ')).explode('Stimuli').reset_index()
        df_words = df_words.rename(columns={'index': 'text_id', 'Stimuli': 'word', 'Condition': 'condition',
                                            'nounfemmasc': 'noun'})
        return self.df_fixation, df_words

    def assign_pos(self, df, ii, look_for_ent, itarget, target, tokenized_text, text_id):
        if look_for_ent:
            if (
                    itarget == 0 or target.startswith(("-", "##", 'year')) or
                    tokenized_text[itarget + 1].startswith("-") or
                    tokenized_text[itarget] == (("70"))) or \
                    (text_id in np.arange(272, 279) and target == 'doctor'
                    ):
                df.loc[ii, 'pos'] = 'ent'
            else:
                look_for_ent = False
                df.loc[ii, 'pos'] = 'unk'
        elif target in ['he', 'she', self.pronoun]:
            df.loc[ii, 'pos'] = 'pron'
        else:
            df.loc[ii, 'pos'] = 'unk'

        return df, look_for_ent