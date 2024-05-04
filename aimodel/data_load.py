from datasets import load_dataset
import pandas as pd
from torchtext.data.utils import get_tokenizer
from sklearn.model_selection import train_test_split
from torchtext.vocab import build_vocab_from_iterator
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
import pickle
from datasets import load_dataset, Features, ClassLabel, Value, Array2D, Array3D
#Setting tokenizer as global 
tokenizer = get_tokenizer('basic_english')

def load_datasets():
    # Define the features with the correct types
    features = Features({
        'Question': Value('string'),
        'Answer': Value('string')  # or use 'float' if answers are numeric and can include decimals
    })

    # Load the dataset
    dataset = load_dataset(
        'csv',
        data_files={'train': 'Comprehensive_Maths_Dataset_Shuffle.csv'},
        features=features
    )
    subset = dataset['train'].select(range(1500))
    df =  convert_dataset(subset)
    return df

def format_prompt(r):
  #Creating question
  return f'''{r["Question"]}'''


def format_label(r):
  #Creating answer
  return r['Answer']


def convert_dataset(ds):
    prompts = [format_prompt(i) for i in ds]
    labels = [format_label(i) for i in ds]
    df = pd.DataFrame.from_dict({'question': prompts, 'answer': labels})
    return df

def get_train_val(df):
    special_tokens = ['<SOS', '<EOS>', '<UNK>', '<PAD>']
    # create zip from querstion and answer
    qa_pairs = list(zip(df['question'], df['answer']))


    # Split into training and validation sets
    train_pairs, val_pairs = train_test_split(qa_pairs, test_size=0.5, random_state=42)

    # Separate questions and answers for convenience
    train_questions, train_answers = zip(*train_pairs)
    val_questions, val_answers = zip(*val_pairs)
    return build_vocab(train_questions, train_answers,  val_questions, val_answers, df)


# Function to yield list of tokens
def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

# Function to build vocab
def build_vocab(train_questions, train_answers,  val_questions, val_answers, df):
    train_texts = train_questions + train_answers + val_questions + val_answers
    VOCAB = build_vocab_from_iterator(yield_tokens(train_texts), specials=['<PAD>', '<SOS>', '<EOS>', '<UNK>'])
    VOCAB.set_default_index(VOCAB['<UNK>'])
    VOCAB_SIZE = len(VOCAB)
    INPUT_SEQ_LEN = df['question'].str.split().str.len().median().__int__()
    TARGET_SEQ_LEN = df['question'].str.split().str.len().median().__int__()
    print('VOCAB_SIZE:', VOCAB_SIZE)
    print('INPUT_SEQ_LEN:', INPUT_SEQ_LEN)
    print('TARGET_SEQ_LEN:', TARGET_SEQ_LEN)
    return [VOCAB, INPUT_SEQ_LEN, TARGET_SEQ_LEN, VOCAB_SIZE]

def tokens_to_text(tokens, VOCAB):
    # check if token is tensor or numpy array
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.cpu().numpy()
    special_tokens = np.array([VOCAB['<SOS>'], VOCAB['<PAD>'], VOCAB['<UNK>'], VOCAB['<EOS>']])
    tokens = [token for token in tokens if token not in special_tokens]
    return ' '.join(VOCAB.lookup_tokens(tokens))

def text_to_tokens(text, VOCAB):
    return [VOCAB[token] for token in tokenizer(text)]

# Create a custom PyTorch Dataset
class QADataset(Dataset):
    def __init__(self, df, vocab, tokenizer, input_seq_len, target_seq_len):
        self.df = df
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.input_seq_len = input_seq_len
        self.target_seq_len = target_seq_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        question, answer = row['question'], row['answer']

        # Tokenize and encode the sequences
        question_tokens = text_to_tokens(question, self.vocab)
        answer_tokens = text_to_tokens(answer, self.vocab)

        # Pad the sequences
        enc_src = self.pad_sequence(question_tokens + [self.vocab['<EOS>']], self.input_seq_len)
        dec_src = self.pad_sequence([self.vocab['<SOS>']] + answer_tokens, self.target_seq_len)
        trg = self.pad_sequence([self.vocab['<SOS>']] + answer_tokens + [self.vocab['<EOS>']], self.target_seq_len)

        return enc_src, dec_src, trg

    def pad_sequence(self, seq, max_len):
        return F.pad(torch.LongTensor(seq), (0, max_len - len(seq)), mode='constant', value=self.vocab['<PAD>'])
    
def pytorch_dataset_final(df, VOCAB, INPUT_SEQ_LEN, TARGET_SEQ_LEN):
    dataset = QADataset(df, VOCAB, tokenizer, INPUT_SEQ_LEN, TARGET_SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    out = next(iter(dataloader))
    return [out, dataloader]

def save_vocab(VOCAB):
    print("[~] Saving VOCAB for Inference")
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(VOCAB, f)

def main_fire_all_dataset():
    df = load_datasets()
    build_vocab_data = get_train_val(df)
    VOCAB = build_vocab_data[0]
    save_vocab(VOCAB)
    INPUT_SEQ_LEN = build_vocab_data[1]
    TARGET_SEQ_LEN = build_vocab_data[2]
    VOCAB_SIZE = build_vocab_data[3]
    outs = pytorch_dataset_final(df, VOCAB, INPUT_SEQ_LEN, TARGET_SEQ_LEN)
    out = outs[0]
    dataloader = outs[1]
    return out, VOCAB, VOCAB_SIZE, INPUT_SEQ_LEN, TARGET_SEQ_LEN, dataloader