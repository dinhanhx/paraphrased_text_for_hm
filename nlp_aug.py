# https://github.com/makcedward/nlpaug#citing
import nlpaug.augmenter.word as naw
import time
import pandas as pd
from tqdm import tqdm
tqdm.pandas(ascii=True)

data_df = pd.read_json('data_test.jsonl', lines=True)

# https://nlpaug.readthedocs.io/en/latest/augmenter/word/context_word_embs.html
model_path = 'bert-base-cased'
actions = ['insert', 'substitute']

for action in actions:
    # Load model
    ts = time.time()
    aug = naw.ContextualWordEmbsAug(model_path=model_path, action=action)
    print(f'Load model takes {time.time() - ts} seconds')
    print(f'Using {model_path} to {action}')

    # Augment
    ts = time.time()
    data_df['paraphrased_text'] = data_df['text'].progress_apply(lambda x: aug.augment(x))
    print(f'Augmentation takes {time.time() - ts} seconds')

    # Save augmented
    data_df['id'] = data_df['id'].apply(lambda x: str(x).zfill(5))
    out_file = f'data_test_paraphrased_nlpaug_{model_path}_{action}.jsonl'
    data_df.to_json(out_file, orient='records', lines=True)