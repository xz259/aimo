import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import Dataset
from transformers import PreTrainedTokenizerFast
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

train = pd.read_csv('/data/train_essays.csv')
test = pd.read_csv('/data/test_essays.csv')

train = train.drop_duplicates(subset=['text'])
train = train.reset_index(drop=True)

def tokenize(df, LOWERCASE=False, VOCAB_SIZE=30522):
    raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    raw_tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFC()] + [normalizers.Lowercase()] if LOWERCASE else []
    )
    raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens)

    dataset = Dataset.from_pandas(train[['text']])

    def train_corp_iter():
        for i in range(0, len(dataset), 1000):
            yield dataset[i : i + 1000]["text"]

    raw_tokenizer.train_from_iterator(train_corp_iter(), trainer=trainer)
    
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )


    tokenized_texts = [tokenizer.tokenize(text) for text in df['text'].tolist()]
    

    return tokenized_texts

bpe_train = tokenize(train)
bpe_test = tokenize(test)

def dummy(text):
    return text

def vectorize(tokenized_texts):
    
    vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, 
                                 analyzer='word', tokenizer=dummy, preprocessor=dummy, 
                                 token_pattern=None, strip_accents='unicode', min_df=2, max_features=5000000)


    X_matrix = vectorizer.fit_transform(tokenized_texts)


    return X_matrix

X_train = vectorize(bpe_train)
X_test = vectorize(bpe_test)

from scipy.sparse import save_npz, load_npz

save_npz('/data/processed_train.npz', X_train)
save_npz('/data/processed_test.npz', X_test)

X_train = load_npz('/data/processed_train.npz')
X_test = load_npz('/data/processed_test.npz')
