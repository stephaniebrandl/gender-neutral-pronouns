import yaml
from os import path, makedirs
import pickle


def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def create_dir(path_out):
    if not path.isdir(path_out):
        makedirs(path_out)


def dump_cloze_files(lan, pronoun, df_cloze, hidden=None):
    create_dir(f'./data/cloze_files/{lan}')
    df_cloze.to_pickle(f'./data/cloze_files/{lan}/df_cloze_{pronoun}.pkl')
    pickle.dump(hidden, open(f"./data/cloze_files/{lan}/hidden_{pronoun}.p", "wb"))


def load_cloze_files(lan, pronoun):
    df_cloze = pickle.load(open(f'./data/cloze_files/{lan}/df_cloze_{pronoun}.pkl', 'rb'))
    hidden = pickle.load(open(f"./data/cloze_files/{lan}/hidden_{pronoun}.p", "rb"))
    return df_cloze, hidden

