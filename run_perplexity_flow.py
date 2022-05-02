from eyetracking.utils import read_yaml, dump_cloze_files, load_cloze_files
from eyetracking.data import SWEyetracking, DAEyetracking, ENEyetracking
from eyetracking.cloze_scores import proc
import pickle
import numpy as np
from os.path import join
from utils import extract_attentions
from scipy.stats import spearmanr

config = read_yaml('config.yaml')
lan = config['language']
pron = config['pronoun']

if lan == 'en':
    data = ENEyetracking(pronoun=pron)
elif lan == 'da':
    data = DAEyetracking(pronoun=pron)
elif lan == 'sv':
    data = SWEyetracking(path=config['data_path']['eyetracking'])

df_fixation, df_words = data.preprocess_df()

try:
    df_cloze, _ = load_cloze_files(lan, pron)
except FileNotFoundError:
    df_cloze, _ = proc(df_words, data, masked=True)
    dump_cloze_files(lan, pron, df_cloze, hidden=None)

try:
    layers = [0, 5, 11]
    flow = []
    for id in range(384):
        _, tokens = pickle.load(open(join(config['data_path']['flow'].format(lan, pron), f"flow{id}_0.pkl"), "rb"))
        flow_tmp = np.zeros([1, 3, len(tokens[1:-1]), len(tokens[1:-1])])
        for ii, layer in enumerate(layers):
            tmp, _ = pickle.load(open(join(config['data_path']['flow'].format(lan, pron), f"flow{id}_{layer}.pkl"), "rb"))
            flow_tmp[0, ii] = tmp[1:-1, 1:-1]
        flow.append(flow_tmp)
        tokens.append(tokens)

except FileNotFoundError:
    print('matrix with flow values not loaded, please check paths')

link, perpl = extract_attentions(df_cloze, flow, 'pronoun')

for ipronoun in [-1, 1]:
    for ii, ilayer in enumerate(layers):
        print(f"correlation for {data.dicts['pronoun'][ipronoun]} in layer {ilayer}: ",
              np.around(spearmanr(perpl[ipronoun], link[ipronoun][:, ii])[0], decimals=2))
    print('\n')

print('relative increase in perplexity: ', np.around(np.mean(perpl[1]) / np.mean(perpl[-1]), decimals=2))