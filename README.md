# Maths-AI-Algo-Task
Maths-AI-Algo


<h1 align="center">
  <br>
  <a href=""><img src="https://github.com/Shivangx01b/Maths-AI-Algo-Task/blob/main/static/Maths_AI_Algo_Logo.png" alt="" width="1200px;"></a>
  <br>
</h1>

<h2 align="center">
  <br>
  <a href=""><img src="https://github.com/Shivangx01b/Maths-AI-Algo-Task/blob/main/static/logo.png" alt="" width="1200px;"></a>
  <br>
</h2>

# How to run the code 

- Note: Docker not added a build for ai bot will be huge ... for cuda based system

- Step 1
  ```
  pip install -r requirements.txt
  ```
- Step 2
  ```
  export OPENAI_API_KEY="<your gpt4 keys"
  ```
- Step 3
  ```
  uvicorn main:app --reload
  ```
- Step 4
    - a) Call full llm based solution
         - Request
             ```
             curl --location 'http://localhost:8000/query-llm' \
                                --header 'Content-Type: application/json' \
                                --data '{"useranswer":"20 and 18", "conversation": "'\''[{'\''user'\'': '\''I am in 7th grade'\'', '\''bot'\'': '\''Hi there! I'\''m Zoe, your AI-Tutor, ready to help you out! 🌟 What subject or topic can I assist you with today?'\'', '\''date'\'': '\''2024-02-01T03:15:48.962Z'\''}, {'\''user'\'': '\''I live in California'\'', '\''bot'\'': '\''Awesome! California is a great place. 🌴 What'\''s the topic or subject you'\''re working on?'\'', '\''date'\'': '\''2024-02-01T03:16:13.529Z'\''}, {'\''user'\'': '\''Common core'\'', '\''bot'\'': '\''Got it! Common Core has lots of areas. Which part are you focusing on? Math, English, or something else? 😊'\'', '\''date'\'': '\''2024-02-01T03:16:44.316Z'\''}, {'\''user'\'': '\''Math'\'', '\''bot'\'': '\''Great, I love math! 🧮 What specific math concept are you studying right now?'\'', '\''date'\'': '\''2024-02-01T03:16:54.172Z'\''}, {'\''user'\'': [{'\''type'\'': '\''text'\'', '\''text'\'': '\'''\''}, {'\''type'\'': '\''image_url'\'', '\''image_url'\'': {'\''url'\'': '\''https://zoe-images.s3.amazonaws.com/file_415_1706757494.jpg'\''}}], '\''bot'\'': '\''Looks like we'\''re working with algebra! Have you tried setting up an equation for this problem yet? 🤔'\'', '\''date'\'': '\''2024-02-01T03:18:20.759Z'\''}, {'\''user'\'': '\''No'\'', '\''bot'\'': '\''No problem! Let'\''s start by identifying what '\''x'\'' represents in this scenario. Any ideas? 🕵️\u200d♀️✨'\'', '\''date'\'': '\''2024-02-01T03:18:37.870Z'\''}, {'\''user'\'': '\''68'\'', '\''bot'\'': '\''I see you mentioned the number 68. If we'\''re setting up the equation for the problem, '\''x'\'' would represent Daniel'\''s spending. Tracy spent $38 less than 3 times what Daniel spent. Can you try to express this as an equation using '\''x'\''? 🧐💡'\'', '\''date'\'': '\''2024-02-01T03:19:54.779Z'\''}, {'\''user'\'': '\''3x-38'\'', '\''bot'\'': '\''That'\''s a great start! Now, how can we solve for '\''x'\'' in the equation you'\''ve written? 🤔🔍'\'', '\''date'\'': '\''2024-02-01T03:20:24.326Z'\''}]'\''"}'
            ```

        - Response
            ```
                      {
              "Evaluation": "NOT_EQUIVALENT",
              "original_question_got": "If Tracy spent $38 less than 3 times what Daniel spent and Tracy's spending is represented by the equation 3x - 38, how can we solve for 'x'?",
              "botanswer": "To solve for 'x' in the equation 3x - 38, we need to know the amount Tracy spent. Without this information, it is not possible to determine the value of 'x'.",
              "useranswer": "20 and 18"
          
          }          
            ```
            
    - b) Call transformer trained model
        - Request
            ```
              curl --location 'http://localhost:8000/query-ai-bot' \
                    --header 'Content-Type: application/json' \
                    --data '{"useranswer":"2196", "conversation": "'\''[{'\''user'\'': '\''I don'\''t want to do math'\'', '\''bot'\'': '\''Hey there! 🌟 I'\''m Zoe, your friendly AI-Tutor, here to make math a bit more fun. 😊 What grade and state are you studying in? Let'\''s tackle some math together! What topic are you on?'\'', '\''date'\'': '\''2024-03-01T18:11:28.267Z'\''}, {'\''user'\'': '\''I hate math'\'', '\''bot'\'': '\''Oh no, hate is such a strong word! 😅 But hey, no pressure. Maybe we can find a way to make it a bit more enjoyable. What part of math is giving you trouble? Let'\''s chat about it! 🤓'\'', '\''date'\'': '\''2024-03-01T18:11:46.878Z'\''}, {'\''user'\'': '\''what is 61 times 36 ?'\'', '\''bot'\'': '\''Sure, to find that, you multiply 61 by 36. What do you get? 🧮'\'', '\''date'\'': '\''2024-03-01T18:12:30.740Z'\''}, {'\''user'\'': '\''But why 36?'\'', '\''bot'\'': '\''Beacause it says 62 time 36, so it means multiply'\'', '\''date'\'': '\''2024-03-01T18:12:48.001Z'\''}]'\''"}'
            ```
        - Response
            ```
                          {
              "Evaluation": "EQUIVALENT",
              "original_question_got": "What is 61 times 36?",
              "botanswer": "2196",
              "useranswer": "2196"
          }
            ```



# Explaning two Solutions

## Why two different solution (My thought process)

- We had to create a AI based system which can understand the conversation of user and bot
    - Answer: Can make parser to fetch image and history, Can use gpt4 for parsing
- Can do computation
    - Answer: Can train a transformer based model (becasuse transformers can understand complex relations) or can use gpt4 based agent with specific prompt to give it understanding. 
- Can verify the the answer
    - Answer: Can use gpt4 agent to veirfy the trained AI answer and user's answer
- Accuracy should be really good
    - Answer: Trained model (Transformer) when trained and infrenced with GPU can be good and fast.



## Solution 1 using only GPT4

- We have 5 agents

- One agent for doing pre flight checks to see if any image found -> feed to gpt 4 vision to get question back

- Once image if found -> Send to agent 2 to convert the whole converstation to original maths question

- Then next third agent to solve maths

- Then last agent to cross check the input by user and solution given by third agent -> Mark as EQUIVALENT or NOT_EQUIVALENT


- Way to Improve
  -  Refine prompt given to each agent
  -  Using Langraph for better agent decisions
  -  Make parser to feect image and converstation instead of using GPT4


## Solution 2 using Transformer trained Model with Gpt 4 for parsing

- We have trained a Transformer based model using pytorch (Please refer to 'Explaning How this transformer bot works' to understand how this works)
  
- We take one agent or doing pre flight checks to see if any image found -> feed to gpt 4 vision to get question back
  
- Once image if found -> Send to agent 2 to convert the whole converstation to original maths question

- Then we feed this question to our trained model -> Get answer

- Then last agent to cross check the input by user and solution given by third agent -> Mark as EQUIVALENT or NOT_EQUIVALENT

- Way to Improve
  -  Train on more dataset, currently data is is only 1500 random maths questions
  -  Use MultiHead Attention then SelfAttention
  -  Using best model saving in each epoch of training
  -  Make parser to feect image and converstation instead of using GPT4

# Explaning How this transformer bot works

## Contents
- Introduction

- Making dataset
    - Getting data
    - Create vocabulary
    - Creating Pytorch dataloader

- Model
    - Creating model classes
    - How it all works Together

- Training
    - Calling Transformer
    - Making traning loop
    - Saving

- Inference
    - Loading Model
    - Chatbot Inference

## Introduction
The aim for the document is to provide you how a Transformer based chatbot can be create with Pytorch to help user's answer their maths realted questions.
Will explore how we can do that in few steps with a power full Transfomer layer provide by Pytorch but before that we will have a dataset, will pre process it as per our needs and will create differenct Layers of Transfomers and will pack all of this together to train a model. Later will make a inference so that our user's can interact with model. So let's get started !

## Making dataset
Traning a model is not a very big deal as with view lines of python code any one can do stuffs which would feel like magic for a non coders.
But getting a dataset for any training a model is the toughest job.
Will create a simple python code which can generate few maths topics questions and answers

#### Getting data
Will create a simple python code which can generate few maths topics questions and answers




```python
import pandas as pd
import random

# Define a wide range of operations including arithmetic, geometry, algebra, calculus, and probability with correct placeholders
comprehensive_operations = [
    # Basic Arithmetic
    ("What is {} plus {}?", lambda x, y: x + y),
    ("What is {} minus {}?", lambda x, y: x - y),
    ("What is {} times {}?", lambda x, y: x * y),
    ("What is {} divided by {}?", lambda x, y: x // y),
    # Geometry
    ("What is the perimeter of a square with side length {}?", lambda x: 4 * x),
    ("What is the area of a rectangle with width {} and height {}?", lambda x, y: x * y),
    ("What is the area of a circle with radius {}?", lambda x: round(3.14 * x * x, 2)),
    # Algebra
    ("Solve for x in the equation x + {} = {}.", lambda x, y: y - x),
    ("What is the solution for x in the equation 2x - {} = {}?", lambda x, y: (y + x) // 2),
    # Calculus
    ("Find the derivative of {}x^2 at x = {}.", lambda x, y: 2 * x * y),
    ("Calculate the integral of {}x from 0 to {}.", lambda x, y: (x * y**2) // 2),
    # Probability
    ("What is the probability of rolling a die and getting a number greater than {}?", lambda x: f"{round((6 - x) / 6 * 100, 2)}%"),
    ("A bag contains {} red and {} blue balls. What is the probability of drawing a red ball?", lambda x, y: f"{round(x / (x + y) * 100, 2)}%"),
    # Statistics
    ("Calculate the average of the numbers {}, {}, and {}.", lambda x, y, z: (x + y + z) // 3),
    ("Find the median of {}, {}, and {}.", lambda x, y, z: sorted([x, y, z])[1])
]

# Generate questions and answers
full_questions = []
full_answers = []

for operation in comprehensive_operations:
    op_text, func = operation
    for _ in range(100):  # Generating 10,000 questions for each operation type
        nums = [random.randint(1, 100) for _ in range(op_text.count("{}"))]  # Generate random numbers for each placeholder
        
        question = op_text.format(*nums)  # Format question with numbers
        answer = str(func(*nums))  # Calculate answer using the operation function
        
        full_questions.append(question)
        full_answers.append(answer)

# Creating DataFrame from the questions and answers
full_dataset = pd.DataFrame({
    "Question": full_questions,
    "Answer": full_answers
})

# Shuffle the DataFrame
full_dataset = full_dataset.sample(frac=1).reset_index(drop=True)

# Save the dataset to a CSV file
full_dataset_path = "Comprehensive_Maths_Dataset_Shuffle.csv"
full_dataset.to_csv(full_dataset_path, index=False)
```


Get the features

```python
from datasets import load_dataset, Features, ClassLabel, Value, Array2D, Array3D

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

# Now access the features and the data
print(dataset['train'].features)
print(dataset['train'][0])  # Print the first row to check
```

Get a subset

```python
subset = dataset['train'].select(range(1500))
```

Convert to a prompt required
```python
def format_prompt(r):
  return f'''{r["Question"]}'''


def format_label(r):
  return r['Answer']


def convert_dataset(ds):
    prompts = [format_prompt(i) for i in ds]
    labels = [format_label(i) for i in ds]
    df = pd.DataFrame.from_dict({'question': prompts, 'answer': labels})
    return df


df_train = convert_dataset(subset)
df_train

```



#### Create vocabulary
Now that we have a dataset, now let's have a train, test split and build a vocabulary. 



```python
def get_train_val(df):
    special_tokens = ['<SOS', '<EOS>', '<UNK>', '<PAD>']
    # create zip from querstion and answer
    qa_pairs = list(zip(df['question'], df['answer']))


    # Split into training and validation sets
    train_pairs, val_pairs = train_test_split(qa_pairs, test_size=0.5, random_state=42)

    # Separate questions and answers for convenience
    train_questions, train_answers = zip(*train_pairs)
    val_questions, val_answers = zip(*val_pairs)
```
We will pass the df to the get_train_val which will give us the train_questions,  train_answers, val_questions and val_answers

Now let's  pass these to the following function which will build vocabulary for us 

```python
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
```

It uses build_vocab_from_iterator from Pytorch and add special  tokens to give us the VOCAB (variable) which we will use later for our model.

#### Creating Pytorch dataloader

Now that we have a vocabulary let's create a Pytorch dataloader, before that we need a way to tokenize out data (question or answer) and have a way to get it back, that is token -> text and text -> tokens

For that let's create few functions

```python
#Setting tokenizer as global 
tokenizer = get_tokenizer('basic_english')

def tokens_to_text(tokens, VOCAB):
    # check if token is tensor or numpy array
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.cpu().numpy()
    special_tokens = np.array([VOCAB['<SOS>'], VOCAB['<PAD>'], VOCAB['<UNK>'], VOCAB['<EOS>']])
    tokens = [token for token in tokens if token not in special_tokens]
    return ' '.join(VOCAB.lookup_tokens(tokens))

def text_to_tokens(text, VOCAB):
    return [VOCAB[token] for token in tokenizer(text)]
```

With these functions we can have our  token -> text and text -> tokens.

Now let's create a custom Pytorch dataset

```python
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
```
The above function will take df, vocab, tokenizer, input_seq_len, target_seq_len ( input_seq_len, target_seq_len from build vocab function) and will tokenize questions and answers then it will pad the tokens  for a source in this case enc_src and dec rc and will have a target.

Now will call the QADataset calls and will pass tha values to Dataloader from Pytroch to shuffle.

```python
def pytorch_dataset_final(df, VOCAB, INPUT_SEQ_LEN, TARGET_SEQ_LEN):
    dataset = QADataset(df, VOCAB, tokenizer, INPUT_SEQ_LEN, TARGET_SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    out = next(iter(dataloader))
    return [out, dataloader]
```

Now that we have eveything. We can chain all these function together to work as a single unit. Like,

```python
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

```

- Note:  I made a save_vocab() function to use vocab in inference mode once the model is done training.


## Model
We will now create few model classes which will be called from the Transfomer Class of Pytroch to help us train the complex relations between question and answers.
We will define the classes and will explain how the inputs and functions work, then later will understand how these classes interact with one another as a single unit

#### Creating model classes
Let's start by creating a self attention layer which we can do like,

```python
import math
import torch
import torch.nn as nn
class SelfAttention(nn.Module):
    def __init__(self, emb_size, heads):
        super(SelfAttention, self).__init__()
        self.emb_size = emb_size
        self.heads = heads
        self.head_dim = emb_size // heads

        assert (self.head_dim * heads == emb_size), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, emb_size)

    def forward(self, values, keys, query, mask):
        batch_size = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(batch_size, value_len, self.heads, self.head_dim)
        keys = keys.reshape(batch_size, key_len, self.heads, self.head_dim)
        queries = query.reshape(batch_size, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Calculate energy
        # energy = torch.matmul(queries, keys.transpose(1, 2)) # (batch_size, head, query_len, key_len)')
        energy = torch.einsum('bqhd,bkhd->bhqk', [queries, keys]) # (batch_size, head, query_len, key_len)')

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        d_k = keys.shape[3]
        attention = torch.softmax(energy / (d_k ** 0.5), dim=3) # (batch_size, head, query_len, key_len)

        out = torch.einsum('bhqk,bvhd->bqhd', [attention, values]) # (batch_size, query_len, head, embed_dim)
        out = out.reshape(batch_size, query_len, self.heads * self.head_dim) # (batch_size, query_len, embed_dim)
        out = self.fc_out(out) # (batch_size, query_len, embed_dim)
        return out
```
This Python class defines a `SelfAttention` module, which is a component used in transformer models, a type of deep learning model commonly used in natural language processing. Let's break it down line by line:

1. `class SelfAttention(nn.Module):` - This line defines a new class called `SelfAttention` which inherits from `nn.Module`, a base class for all neural network modules in PyTorch. This inheritance means `SelfAttention` will have all the functionalities of a PyTorch module.
    
2. `def __init__(self, emb_size, heads):` - This is the constructor for the `SelfAttention` class. It initializes the module with two parameters: `emb_size` (embedding size) and `heads` (number of attention heads).
    
3. `super(SelfAttention, self).__init__()` - This line calls the constructor of the parent class (`nn.Module`), which is necessary for proper PyTorch module initialization.
    
4. `self.emb_size = emb_size` - Here, the embedding size is stored as an instance variable.
    
5. `self.heads = heads` - The number of attention heads is stored as an instance variable.
    
6. `self.head_dim = emb_size // heads` - This calculates the dimensionality of each attention head. The embedding size is divided by the number of heads to ensure that the embeddings are split evenly across the heads.
    
7. `assert (self.head_dim * heads == emb_size), "Embedding size needs to be divisible by heads"` - This line ensures that the embedding size is divisible by the number of heads. It's a safety check to prevent configuration errors.
    
8. `self.values`, `self.keys`, `self.queries` - These lines create linear transformations for the values, keys, and queries respectively. These transformations are used to project the inputs into different spaces for the attention mechanism. Note that `bias=False` means these linear layers do not have an additive bias.
    
9. `self.fc_out = nn.Linear(heads * self.head_dim, emb_size)` - This is a linear layer to transform the concatenated outputs of the attention heads back to the original embedding dimension.
    
10. `def forward(self, values, keys, query, mask):` - This defines the forward pass of the module. It takes four arguments: `values`, `keys`, `query`, and an optional `mask`.
    
11. `batch_size = query.shape[0]` - This gets the batch size from the shape of the query.
    
12. `value_len, key_len, query_len = ...` - Extracts the sequence lengths of the values, keys, and query.
    
13. The next few lines reshape these inputs and apply the linear transformations defined earlier.
    
14. `energy = torch.einsum('bqhd,bkhd->bhqk', [queries, keys])` - This computes the attention scores (or "energy") using Einstein summation convention. This effectively calculates the dot product of queries and keys.
    
15. `if mask is not None:` - This applies a mask to the attention scores if provided, useful for handling variable-length sequences.
    
16. `attention = torch.softmax(energy / (d_k ** 0.5), dim=3)` - This computes the attention weights using softmax, scaling by the square root of the dimension of the keys.
    
17. `out = torch.einsum('bhqk,bvhd->bqhd', [attention, values])` - This calculates the weighted sum of the values based on the attention weights.
    
18. `out = out.reshape(batch_size, query_len, self.heads * self.head_dim)` and `out = self.fc_out(out)` - These lines reshape the output and apply the final linear transformation.
    
19. `return out` - Finally, the output of the self-attention mechanism is returned.
    

This class is a key component in transformer architectures, allowing the model to focus on different parts of the input sequence, which is crucial for tasks like language modeling and machine translation.

Now let's create a TransfomerBlock, like

```python
class TransformerBlock(nn.Module):
    def __init__(self, emb_size, heads, forward_expansion, drop_out) -> None:
        super(TransformerBlock, self).__init__()

        self.attention = SelfAttention(emb_size, heads)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(in_features=emb_size, out_features=forward_expansion * emb_size, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=forward_expansion * emb_size, out_features=emb_size, bias=True),
        )

        self.dropout = nn.Dropout(drop_out)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        norm = self.norm1(attention + query)
        norm = self.dropout(norm)

        forward = self.feed_forward(norm)
        out = self.norm2(forward + norm)
        return out
```
This Python class defines a `TransformerBlock`, which is a component of a Transformer model, widely used in modern natural language processing. Let's break it down line by line:

1. `class TransformerBlock(nn.Module):` - This line declares a new class named `TransformerBlock` which inherits from `nn.Module`, the base class for all neural network modules in PyTorch.
    
2. `def __init__(self, emb_size, heads, forward_expansion, drop_out) -> None:` - This is the constructor for the `TransformerBlock` class. It initializes the block with four parameters: `emb_size` (embedding size), `heads` (number of attention heads), `forward_expansion` (a factor for expanding dimensions in the feed-forward layer), and `drop_out` (dropout rate).
    
3. `super(TransformerBlock, self).__init__()` - Calls the constructor of the parent class (`nn.Module`) for proper initialization.
    
4. `self.attention = SelfAttention(emb_size, heads)` - Creates an instance of the `SelfAttention` class defined earlier, initializing it with the embedding size and the number of heads.
    
5. `self.norm1 = nn.LayerNorm(emb_size)` and `self.norm2 = nn.LayerNorm(emb_size)` - These lines create two layer normalization modules. Layer normalization is used in Transformers for stabilizing the learning process.
    
6. `self.feed_forward = nn.Sequential(...)` - Defines a feed-forward neural network which is part of the Transformer block. This network expands the dimensions of its inputs by a factor of `forward_expansion`, applies a ReLU activation, and then projects it back to the original dimension.
    
7. `self.dropout = nn.Dropout(drop_out)` - Initializes a dropout layer with the specified dropout rate. Dropout is a regularization technique used to prevent overfitting.
    
8. `def forward(self, value, key, query, mask):` - This method defines the forward pass of the Transformer block. It takes four arguments: `value`, `key`, `query`, and `mask`.
    
9. `attention = self.attention(value, key, query, mask)` - This line computes the self-attention for the input values, keys, and queries, applying the optional mask if provided.
    
10. `norm = self.norm1(attention + query)` - Applies the first layer normalization after adding a skip connection (residual connection) from the query. This is part of the 'Add & Norm' step in the Transformer architecture.
    
11. `norm = self.dropout(norm)` - Applies dropout to the normalized output for regularization.
    
12. `forward = self.feed_forward(norm)` - Passes the output through the feed-forward network.
    
13. `out = self.norm2(forward + norm)` - Again applies layer normalization after adding another skip connection. This completes the second 'Add & Norm' step in the Transformer block.
    
14. `return out` - Returns the output of the Transformer block.
    

The `TransformerBlock` is a fundamental building block of Transformer models. It consists of a self-attention layer followed by a feed-forward neural network, with layer normalization and skip connections employed at each step. This architecture is crucial for handling long-range dependencies and complex patterns in sequences, making it highly effective for tasks like language translation, text generation, and many others in the field of NLP.


Now let's create WordPosition Embedding

```python
class WordPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, max_seq_len, emb_size, device, fixed=True):
        super(WordPositionEmbedding, self).__init__()

        self.device = device
        self.fixed = fixed

        self.word_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size, device=device)

        if fixed:
            # Create fixed (non-learnable) position embeddings
            position = torch.arange(max_seq_len).unsqueeze(1) #[max_seq_len, 1]
            div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size))
            position_embedding = torch.zeros(max_seq_len, emb_size)
            position_embedding[:, 0::2] = torch.sin(position * div_term)
            position_embedding[:, 1::2] = torch.cos(position * div_term)
            # Register position_embedding as a buffer
            self.register_buffer('position_embedding', position_embedding)
        else:
            self.position_embedding = nn.Embedding(max_seq_len, emb_size)

    def forward(self, X):
        batch_size, seq_len = X.shape

        # Get word embeddings
        word = self.word_embedding(X)

        if self.fixed:
            # Use fixed position embeddings
            position = self.position_embedding[:seq_len, :]
        else:
            # Get position embeddings
            position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, seq_len)
            position = self.position_embedding(position_ids)

        # Add word and position embeddings
        embeddings = word + position
        return embeddings
```

This Python class defines `WordPositionEmbedding`, a module that generates embeddings for words and their positions in a sequence, commonly used in Transformer models for natural language processing. Let's break it down line by line:

1. `class WordPositionEmbedding(nn.Module):` - This line defines a new class named `WordPositionEmbedding` which inherits from `nn.Module`, the base class for all neural network modules in PyTorch.
    
2. `def __init__(self, vocab_size, max_seq_len, emb_size, device, fixed=True):` - This is the constructor for the `WordPositionEmbedding` class. It initializes the module with five parameters: `vocab_size` (size of the vocabulary), `max_seq_len` (maximum length of sequences), `emb_size` (embedding size), `device` (the device to run the model on, e.g., CPU or GPU), and `fixed` (a boolean indicating if position embeddings are fixed or learnable).
    
3. `super(WordPositionEmbedding, self).__init__()` - Calls the constructor of the parent class (`nn.Module`) for proper initialization.
    
4. `self.device = device` and `self.fixed = fixed` - Stores the `device` and `fixed` flag as instance variables.
    
5. `self.word_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size, device=device)` - Initializes a word embedding layer which maps each word in the vocabulary to a high-dimensional vector.
    
6. `if fixed:` - This conditional checks if fixed (non-learnable) position embeddings are to be used.
    
7. In the `if` block:
    
    - `position = torch.arange(max_seq_len).unsqueeze(1)` - Generates a tensor of shape `[max_seq_len, 1]` containing a sequence of integers from `0` to `max_seq_len-1`.
    - `div_term = torch.exp(...)` - Calculates a term used to vary the wavelength of the sine and cosine functions.
    - `position_embedding = torch.zeros(max_seq_len, emb_size)` - Creates a tensor for storing position embeddings.
    - The following two lines fill the position embedding tensor with sine and cosine values, creating a pattern that the model can use to learn the relative or absolute position of words in a sequence.
8. `self.register_buffer('position_embedding', position_embedding)` - Registers `position_embedding` as a buffer in PyTorch, meaning it is not a model parameter and does not get updated during training.
    
9. `else:` - The `else` block is executed if learnable position embeddings are to be used.
    
    - `self.position_embedding = nn.Embedding(max_seq_len, emb_size)` - Initializes a learnable position embedding layer.
10. `def forward(self, X):` - Defines the forward pass for the embedding module.
    
11. `batch_size, seq_len = X.shape` - Extracts the batch size and sequence length from the input tensor `X`.
    
12. `word = self.word_embedding(X)` - Fetches the word embeddings for each token in the input sequences.
    
13. `if self.fixed:` and the corresponding `else:` - These blocks handle the addition of position embeddings based on whether they are fixed or learnable.
    
    - For fixed embeddings, a slice of the precomputed position embeddings is used.
    - For learnable embeddings, position ids are generated dynamically and passed through the position embedding layer.
14. `embeddings = word + position` - Adds the word and position embeddings element-wise.
    
15. `return embeddings` - Returns the combined embeddings.
    

The `WordPositionEmbedding` class is an essential part of Transformer-based models, allowing them to understand both the meaning of individual words and their position in a sequence, which is crucial for many language understanding tasks.


Now let's create Encoder, like

```python
class Encoder(nn.Module):
    def __init__(self, vocab_size, seq_len, emb_size, n_layers, heads, forward_expansion, drop_out, device):
        super(Encoder, self).__init__()

        self.emb_size = emb_size
        self.device = device

        self.embedding = WordPositionEmbedding(vocab_size, seq_len, emb_size, device, fixed=True)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(emb_size, heads, forward_expansion, drop_out) for _ in range(n_layers)
            ]
        )

        self.dropout = nn.Dropout(drop_out)

    def forward(self, X, mask):
        batch_size, seq_len = X.shape
        out = self.dropout(self.embedding(X))

        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out
```
The `Encoder` class is a core component of a Transformer model, specifically designed for encoding sequences (like sentences in natural language processing tasks). Let's go through its implementation line by line:

1. `class Encoder(nn.Module):` - This line defines the `Encoder` class, which inherits from `nn.Module`, the base class for all neural network modules in PyTorch.
    
2. `def __init__(self, vocab_size, seq_len, emb_size, n_layers, heads, forward_expansion, drop_out, device):` - The constructor of the `Encoder` class. It initializes the encoder with several parameters:
    
    - `vocab_size`: The size of the vocabulary.
    - `seq_len`: The maximum length of the input sequences.
    - `emb_size`: The size of the embeddings.
    - `n_layers`: The number of transformer layers to be used in the encoder.
    - `heads`: The number of attention heads in each transformer block.
    - `forward_expansion`: A factor for expanding dimensions in the feed-forward layer of the transformer block.
    - `drop_out`: The dropout rate for regularization.
    - `device`: The device (CPU or GPU) on which the model will run.
3. `super(Encoder, self).__init__()` - Calls the constructor of the parent class (`nn.Module`), necessary for PyTorch modules.
    
4. `self.emb_size = emb_size` and `self.device = device` - Stores the embedding size and device as instance variables.
    
5. `self.embedding = WordPositionEmbedding(vocab_size, seq_len, emb_size, device, fixed=True)` - Initializes the `WordPositionEmbedding` module, which will create combined word and position embeddings for the input sequences.
    
6. `self.layers = nn.ModuleList([...])` - Creates a list (`nn.ModuleList`) of transformer blocks (`TransformerBlock`). This list contains `n_layers` number of transformer blocks, each initialized with the specified parameters.
    
7. `self.dropout = nn.Dropout(drop_out)` - Initializes a dropout layer with the specified dropout rate, which will be used for regularization.
    
8. `def forward(self, X, mask):` - Defines the forward pass of the encoder. It takes two arguments: `X` (the input data) and `mask` (the attention mask).
    
9. `batch_size, seq_len = X.shape` - Extracts the batch size and sequence length from the input tensor `X`.
    
10. `out = self.dropout(self.embedding(X))` - First, the input `X` is passed through the embedding layer. Then, the dropout is applied to the embeddings.
    
11. `for layer in self.layers:` - Iterates over each transformer block in the encoder.
    
12. `out = layer(out, out, out, mask)` - For each transformer block, the same `out` tensor is used as the value, key, and query. The attention mask is also passed. This process updates the `out` tensor with the transformed representations.
    
13. `return out` - Returns the final output after passing through all the transformer layers.
    

The `Encoder` class in a Transformer model plays a crucial role in processing input sequences and capturing both their content and positional information through self-attention mechanisms and feed-forward networks. The output of this encoder can then be used for various tasks like classification, translation, or feeding into a decoder in a sequence-to-sequence model.

Now let's create DecoderBlock, like

```python
class DecoderBlock(nn.Module):
    def __init__(self, emb_size, heads, forward_expansion, drop_out) -> None:
        super(DecoderBlock, self).__init__()

        self.attention = SelfAttention(emb_size, heads)
        self.norm = nn.LayerNorm(emb_size)
        self.transformer_block = TransformerBlock(emb_size, heads, forward_expansion, drop_out)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, X, value, key, src_mask, trg_mask):
        attention = self.attention(X, X, X, trg_mask)
        query = self.dropout(self.norm(attention + X))
        out = self.transformer_block(value, key, query, src_mask)
        return out
```
The `DecoderBlock` class is an essential component of the decoder part in a Transformer model, typically used in sequence-to-sequence tasks like machine translation. It processes the output of the encoder and generates the target sequence. Let's analyze it line by line:

1. `class DecoderBlock(nn.Module):` - This line declares a new class named `DecoderBlock`, which inherits from `nn.Module`, the base class for all neural network modules in PyTorch.
    
2. `def __init__(self, emb_size, heads, forward_expansion, drop_out) -> None:` - The constructor for the `DecoderBlock` class. It initializes the block with four parameters: `emb_size` (embedding size), `heads` (number of attention heads), `forward_expansion` (a factor for expanding dimensions in the feed-forward layer), and `drop_out` (dropout rate).
    
3. `super(DecoderBlock, self).__init__()` - Calls the constructor of the parent class (`nn.Module`) for proper initialization.
    
4. `self.attention = SelfAttention(emb_size, heads)` - Initializes a self-attention mechanism. This attention mechanism is used to focus on different parts of the input sequence.
    
5. `self.norm = nn.LayerNorm(emb_size)` - Initializes a layer normalization module. Layer normalization is used in Transformers for stabilizing the learning process.
    
6. `self.transformer_block = TransformerBlock(emb_size, heads, forward_expansion, drop_out)` - Initializes a `TransformerBlock`, which includes another self-attention mechanism and a feed-forward network. This block processes the output of the decoder's self-attention layer.
    
7. `self.dropout = nn.Dropout(drop_out)` - Initializes a dropout layer with the specified dropout rate for regularization.
    
8. `def forward(self, X, value, key, src_mask, trg_mask):` - Defines the forward pass of the decoder block. It takes five arguments: `X` (the input to the decoder block), `value` and `key` (outputs from the encoder), `src_mask` (source mask for the encoder's output), and `trg_mask` (target mask for the decoder's input).
    
9. `attention = self.attention(X, X, X, trg_mask)` - Computes self-attention for the input `X`. The same `X` is used as the value, key, and query. The `trg_mask` is used to prevent the model from attending to future tokens in the target sequence.
    
10. `query = self.dropout(self.norm(attention + X))` - Applies a residual connection (skip connection) by adding the input `X` to the attention output, followed by layer normalization and dropout. This output serves as the query for the next attention layer.
    
11. `out = self.transformer_block(value, key, query, src_mask)` - Passes the outputs from the encoder (`value` and `key`), along with the query from the previous step, to the transformer block. The `src_mask` is used to mask the encoder's outputs.
    
12. `return out` - Returns the output of the decoder block.
    

In a Transformer model, the `DecoderBlock` plays a vital role in generating the target sequence. It uses self-attention to understand the context within the target sequence and encoder-decoder attention to focus on relevant parts of the input sequence. The combination of these mechanisms allows the decoder to effectively translate, summarize, or generate text based on the input from the encoder.

Now let's create Transfomer Class from Pytorch,

```python
class TransformerPytroch(nn.Module):
    def __init__(
        self,
        inp_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        emb_size,
        n_layers=1,
        heads=1,
        forward_expansion=1,
        drop_out=0.2,
        max_seq_len=100,
        device=torch.device('cuda')
    ):
        super(TransformerPytroch, self).__init__()

        self.enc_embedding = WordPositionEmbedding(inp_vocab_size, max_seq_len, emb_size, device, fixed=False)
        self.dec_embedding = WordPositionEmbedding(trg_vocab_size, max_seq_len, emb_size, device, fixed=False)


        self.device = device
        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=forward_expansion,
            dropout=drop_out,
            batch_first=True,
            device=device
        )
        self.fc_out = nn.Linear(emb_size, trg_vocab_size)
        self.dropout = nn.Dropout(drop_out)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_src_mask(self, src):
        src_mask = src == self.src_pad_idx

        # (N, src_len)
        return src_mask.to(self.device)

    def forward(self, src, trg):
        batch_size, trg_seq_length = trg.shape

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(
            self.device
        )

        src_emb = self.dropout(self.enc_embedding(src))
        trg_emb = self.dropout(self.dec_embedding(trg))

        out = self.transformer(
            src_emb,
            trg_emb,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask,
        )
        out = self.fc_out(out)
        return out
```

The `TransformerPytroch` class encapsulates a full Transformer model using PyTorch's `nn.Transformer` module, suitable for tasks like machine translation. It includes both an encoder and a decoder along with embedding layers for input and target sequences. Let's analyze it line by line:

1. `class TransformerPytroch(nn.Module):` - Declares the `TransformerPytroch` class, inheriting from PyTorch's `nn.Module`.
    
2. `def __init__(self, inp_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, emb_size, n_layers=1, heads=1, forward_expansion=1, drop_out=0.2, max_seq_len=100, device=torch.device('cuda')):` - The constructor initializes the Transformer model with various parameters:
    
    - `inp_vocab_size` and `trg_vocab_size`: Vocabulary sizes for the input and target languages.
    - `src_pad_idx` and `trg_pad_idx`: Padding indices for the input and target sequences.
    - `emb_size`: Size of the embeddings.
    - `n_layers`: Number of layers in both the encoder and decoder.
    - `heads`: Number of attention heads.
    - `forward_expansion`: A factor for expanding dimensions in the feed-forward layer.
    - `drop_out`: Dropout rate for regularization.
    - `max_seq_len`: Maximum length of the sequences.
    - `device`: The device (CPU or GPU) on which the model will run.
3. `super(TransformerPytroch, self).__init__()` - Initializes the base `nn.Module`.
    
4. `self.enc_embedding = WordPositionEmbedding(...)` and `self.dec_embedding = WordPositionEmbedding(...)` - Initializes word and position embeddings for both the encoder and decoder. Note that `fixed=False` allows these embeddings to be learnable.
    
5. `self.device = device` - Stores the device as an instance variable.
    
6. `self.transformer = nn.Transformer(...)` - Initializes PyTorch's built-in `nn.Transformer` module with specified parameters. It internally contains the encoder and decoder layers, attention mechanisms, and feed-forward networks.
    
7. `self.fc_out = nn.Linear(emb_size, trg_vocab_size)` - A fully connected layer to map the output of the decoder to the target vocabulary size.
    
8. `self.dropout = nn.Dropout(drop_out)` - Initializes a dropout layer.
    
9. `self.src_pad_idx = src_pad_idx` and `self.trg_pad_idx = trg_pad_idx` - Stores the source and target padding indices.
    
10. `def make_src_mask(self, src):` - Defines a method to create a source mask, which will be used to prevent the model from processing padding tokens in the input sequence.
    
11. `src_mask = src == self.src_pad_idx` - Creates the source mask by comparing the input indices to the padding index.
    
12. `return src_mask.to(self.device)` - Returns the source mask, ensuring it's on the correct device.
    
13. `def forward(self, src, trg):` - Defines the forward pass for the model.
    
    - `src` and `trg` are the input and target sequences, respectively.
14. `batch_size, trg_seq_length = trg.shape` - Extracts the target sequence length.
    
15. `src_padding_mask = self.make_src_mask(src)` - Generates the source padding mask.
    
16. `trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(self.device)` - Generates a target mask to prevent the model from 'peeking' at future tokens in the target sequence.
    
17. `src_emb = self.dropout(self.enc_embedding(src))` and `trg_emb = self.dropout(self.dec_embedding(trg))` - Applies embeddings and dropout to the source and target sequences.
    
18. `out = self.transformer(src_emb, trg_emb, src_key_padding_mask=src_padding_mask, tgt_mask=trg_mask,)` - Passes the embedded and masked source and target sequences through the transformer.
    
19. `out = self.fc_out(out)` - Passes the transformer's output through the fully connected layer to obtain the final output.
    
20. `return out` - Returns the final output.
    

The `TransformerPytroch` class provides a complete implementation of a Transformer model using PyTorch's high-level APIs. It integrates embeddings, masking, and the transformer mechanism to process sequences for tasks like translation or text generation.


#### How it all works Together

1. **WordPositionEmbedding**:
    
    - Both the Encoder and Decoder start with this component.
    - For the Encoder, it takes the input sequence (e.g., a sentence in the source language for a translation task) and converts each word into a vector using word embeddings. It also adds positional embeddings to these word vectors to give the model information about the position of each word in the sequence.
    - The Decoder does a similar thing for the target sequence (e.g., the translated sentence in the target language).
2. **Encoder**:
    
    - Composed of several layers, each containing a TransformerBlock.
    - In each TransformerBlock, the sequence first goes through a SelfAttention mechanism. This allows each word in the input sequence to attend to (or, consider) all other words in the same sequence, which is crucial for understanding the context and relationships between words.
    - After self-attention, the sequence is passed through a series of feed-forward neural networks.
    - The output of the Encoder is a transformed representation of the input sequence, enriched with contextual information.
3. **Decoder**:
    
    - Also composed of several layers, each containing a DecoderBlock.
    - In each DecoderBlock, the process starts with a SelfAttention mechanism, similar to the Encoder. However, it's masked to prevent positions from attending to subsequent positions. This is important during training to prevent the model from 'cheating' by seeing the future output.
    - The DecoderBlock also includes an attention mechanism that attends to the output of the Encoder. This helps the Decoder focus on relevant parts of the input sequence.
    - After these attention processes, the sequence is again passed through feed-forward neural networks.
4. **Interaction Between Encoder and Decoder**:
    
    - The output of the Encoder (which encodes the input sequence) is used by each DecoderBlock. It's part of the attention mechanism that allows the Decoder to focus on different parts of the input sequence while generating the output sequence.
5. **Final Output Layer**:
    
    - The output of the Decoder is passed through a final linear layer (often followed by a softmax function). This layer converts the Decoder's output into a probability distribution over the target vocabulary for each position in the output sequence.
    - This step is where the actual prediction of the next word in the sequence occurs (for each position in the sequence).

In summary, the Transformer model processes the input sequence through its Encoder, enriching it with contextual information. The Decoder then uses this enriched representation, along with its own processing, to generate the output sequence step-by-step. The SelfAttention mechanism in both Encoder and Decoder allows the model to focus on different parts of the sequence, capturing complex relationships between words.


## Training
Now that we have a data, token, vocab and model ready let's use them all together to train a model


#### Calling Transformer

We will call the Transfomer which we created and pass the inputs required to train a model, like

```python
def train_main():
    print("[~] Getting Device to use")
    device = get_device()
    print("[+] Using device {0}\n".format(device))

    print("[~] Getting data frame")
    out_from_dataset_module = main_fire_all_dataset()

    dataset = out_from_dataset_module[0]
    VOCAB = out_from_dataset_module[1]
    VOCAB_SIZE = out_from_dataset_module[2]
    INPUT_SEQ_LEN = out_from_dataset_module[3]
    TARGET_SEQ_LEN = out_from_dataset_module[4]
    dataloader = out_from_dataset_module[5]
    print("[+] Data frame {0}\n".format(dataset))
    pytorch_transformer = TransformerPytroch(
        inp_vocab_size = VOCAB_SIZE,
        trg_vocab_size = VOCAB_SIZE,
        src_pad_idx = VOCAB['<PAD>'],
        trg_pad_idx = VOCAB['<PAD>'],
        emb_size = 512,
        n_layers=1,
        heads=4,
        forward_expansion=4,
        drop_out=0.1,
        max_seq_len=TARGET_SEQ_LEN,
        device=device
    ).to(device)
```
The train_main() function first setups up the device and get the data, tokes and vocab from the dataset functions we createdand then pass those to TransformerPytroch and creats a instance of it

#### Making training Loop

Now let's create functions for training. 


```python
def step(model, enc_src, dec_src, trg, loss_fn, VOCAB, device):
    enc_src = enc_src.to(device)
    dec_src = dec_src.to(device)
    trg = trg.to(device)

    # Forward pass through the model
    logits = model(enc_src, dec_src)

    # Shift so we don't include the SOS token in targets, and remove the last logit to match targets
    logits = logits[:, :-1, :].contiguous()
    trg = trg[:, 1:].contiguous()

    loss = loss_fn(logits.view(-1, logits.shape[-1]), trg.view(-1))
    # loss = loss_fn(logits.permute(0, 2, 1), trg)

    # Calculate accuracy
    non_pad_elements = (trg != VOCAB['<PAD>']).nonzero(as_tuple=True)
    correct_predictions = (logits.argmax(dim=2) == trg).sum().item()
    accuracy = correct_predictions / len(non_pad_elements[0])

    return loss, accuracy

def train_step(model, iterator, optimizer, loss_fn, clip, VOCAB, device):
    model.train()  # Set the model to training mode
    epoch_loss = 0
    epoch_acc = 0

    for i, batch in enumerate(iterator):
        enc_src, dec_src, trg = batch

        # Zero the gradients
        optimizer.zero_grad()

        loss, accuracy = step(model, enc_src, dec_src, trg, loss_fn, VOCAB, device)

        # Backward pass
        loss.backward()

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # Update parameters
        optimizer.step()

        # Accumulate loss and accuracy
        epoch_loss += loss.item()
        epoch_acc += accuracy

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate_step(model, iterator, loss_fn, VOCAB, device):
    model.eval()  # Set the model to evaluation mode
    epoch_loss = 0
    epoch_acc = 0

    with torch.no_grad():  # Disable gradient computation
        for i, batch in enumerate(iterator):
            enc_src, dec_src, trg = batch

            loss, accuracy = step(model, enc_src, dec_src, trg, loss_fn, VOCAB, device)

            # Accumulate loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += accuracy

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def train(model, train_loader, optimizer, loss_fn, clip, epochs, VOCAB, device, val_loader=None):
    for epoch in range(epochs):
        train_loss, train_acc = train_step(model, train_loader, optimizer, loss_fn, clip, VOCAB, device)
        result = f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%'

        if val_loader:
            eval_loss, eval_acc = evaluate_step(model, val_loader, loss_fn, VOCAB, device)
            result += f'|| Eval Loss: {eval_loss:.3f} | Eval Acc: {eval_acc * 100:.2f}%'

        print(f'Epoch: {epoch + 1:02}')
        print(result)

    return model
```
These functions call train() which then calls train_step() which does the training loop then we call evaluate_step() to get Eval Loss and Accuracy.

Now will call this train() functions like

```python
loss_function = torch.nn.CrossEntropyLoss(ignore_index=VOCAB['<PAD>'], reduction='mean')
optimizer = optim.Adam(pytorch_transformer.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

print("[~] Training....")
pytorch_transformer = train(pytorch_transformer, dataloader, optimizer, loss_function, clip=1, epochs=100, VOCAB=VOCAB, device=device)
```

#### Saving Model

Once the training is completed we can save the model like,

```python
print("[~] Saving Model")
model_name = "/application/chatbot/pytorch_transformer_model.pth"
torch.save(pytorch_transformer.state_dict(), model_name)
print("[!] Saved Model as {0}".format(model_name))
```

- Note: All code in a single file for traning looks like,

```python

from chatbot.data_load import main_fire_all_dataset
from chatbot.utils import get_device
from chatbot.model import TransformerPytroch
import torch
import torch.optim as optim
import sys
def train_main():
    print("[~] Getting Device to use")
    device = get_device()
    print("[+] Using device {0}\n".format(device))

    print("[~] Getting data frame")
    out_from_dataset_module = main_fire_all_dataset()

    dataset = out_from_dataset_module[0]
    VOCAB = out_from_dataset_module[1]
    VOCAB_SIZE = out_from_dataset_module[2]
    INPUT_SEQ_LEN = out_from_dataset_module[3]
    TARGET_SEQ_LEN = out_from_dataset_module[4]
    dataloader = out_from_dataset_module[5]
    print("[+] Data frame {0}\n".format(dataset))
    pytorch_transformer = TransformerPytroch(
        inp_vocab_size = VOCAB_SIZE,
        trg_vocab_size = VOCAB_SIZE,
        src_pad_idx = VOCAB['<PAD>'],
        trg_pad_idx = VOCAB['<PAD>'],
        emb_size = 512,
        n_layers=1,
        heads=4,
        forward_expansion=4,
        drop_out=0.1,
        max_seq_len=TARGET_SEQ_LEN,
        device=device
    ).to(device)

    loss_function = torch.nn.CrossEntropyLoss(ignore_index=VOCAB['<PAD>'], reduction='mean')
    optimizer = optim.Adam(pytorch_transformer.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    print("[~] Training....")
    pytorch_transformer = train(pytorch_transformer, dataloader, optimizer, loss_function, clip=1, epochs=100, VOCAB=VOCAB, device=device)

    print("[~] Saving Model")
    model_name = "/application/chatbot/pytorch_transformer_model.pth"
    torch.save(pytorch_transformer.state_dict(), model_name)
    print("[!] Saved Model as {0}".format(model_name))

def step(model, enc_src, dec_src, trg, loss_fn, VOCAB, device):
    enc_src = enc_src.to(device)
    dec_src = dec_src.to(device)
    trg = trg.to(device)

    # Forward pass through the model
    logits = model(enc_src, dec_src)

    # Shift so we don't include the SOS token in targets, and remove the last logit to match targets
    logits = logits[:, :-1, :].contiguous()
    trg = trg[:, 1:].contiguous()

    loss = loss_fn(logits.view(-1, logits.shape[-1]), trg.view(-1))
    # loss = loss_fn(logits.permute(0, 2, 1), trg)

    # Calculate accuracy
    non_pad_elements = (trg != VOCAB['<PAD>']).nonzero(as_tuple=True)
    correct_predictions = (logits.argmax(dim=2) == trg).sum().item()
    accuracy = correct_predictions / len(non_pad_elements[0])

    return loss, accuracy

def train_step(model, iterator, optimizer, loss_fn, clip, VOCAB, device):
    model.train()  # Set the model to training mode
    epoch_loss = 0
    epoch_acc = 0

    for i, batch in enumerate(iterator):
        enc_src, dec_src, trg = batch

        # Zero the gradients
        optimizer.zero_grad()

        loss, accuracy = step(model, enc_src, dec_src, trg, loss_fn, VOCAB, device)

        # Backward pass
        loss.backward()

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # Update parameters
        optimizer.step()

        # Accumulate loss and accuracy
        epoch_loss += loss.item()
        epoch_acc += accuracy

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate_step(model, iterator, loss_fn, VOCAB, device):
    model.eval()  # Set the model to evaluation mode
    epoch_loss = 0
    epoch_acc = 0

    with torch.no_grad():  # Disable gradient computation
        for i, batch in enumerate(iterator):
            enc_src, dec_src, trg = batch

            loss, accuracy = step(model, enc_src, dec_src, trg, loss_fn, VOCAB, device)

            # Accumulate loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += accuracy

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def train(model, train_loader, optimizer, loss_fn, clip, epochs, VOCAB, device, val_loader=None):
    for epoch in range(epochs):
        train_loss, train_acc = train_step(model, train_loader, optimizer, loss_fn, clip, VOCAB, device)
        result = f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%'

        if val_loader:
            eval_loss, eval_acc = evaluate_step(model, val_loader, loss_fn, VOCAB, device)
            result += f'|| Eval Loss: {eval_loss:.3f} | Eval Acc: {eval_acc * 100:.2f}%'

        print(f'Epoch: {epoch + 1:02}')
        print(result)

    return model



if __name__ == "__main__":
    train_main()

```

## Inference
Now that our model is trained and ready let's call, will make few function to make it work like a chatbot

#### Loading Model
Let's first load the model

```python
def init_chatbot():
    print("[~] Loading ChatBot")
    pytorch_load_model  = TransformerPytroch(
    inp_vocab_size = 958,
    trg_vocab_size = 958,
    src_pad_idx = 0,
    trg_pad_idx = 0,
    emb_size = 512,
    n_layers=1,
    heads=4,
    forward_expansion=4,
    drop_out=0.1,
    max_seq_len=10,
    device=device
    ).to(device)
    
    pytorch_load_model.load_state_dict(torch.load('/application/chatbot/pytorch_transformer_model.pth', map_location=torch.device('cpu')))
    return pytorch_load_model
```

Now that our model is loaded let's see how we can have a chatbot inference


#### Chatbot Inference
We can create chatbot inference like

```python
#Loading VOCAB
with open('/application/chatbot/vocab.pkl', 'rb') as f:
    VOCAB = pickle.load(f)
device = get_device()

def prepare_model_input(question, max_length=50):
    # Tokenize the input question
    tokenized_question = text_to_tokens(question, VOCAB)
    enc_src = tokenized_question + [VOCAB['<EOS>']]  # Add SOS and EOS tokens

    # Prepare a placeholder for the decoder's input
    dec_src = torch.LongTensor([VOCAB['<SOS>']]).unsqueeze(0).to(device)

    # Convert to tensor and add batch dimension
    enc_src = F.pad(torch.LongTensor(enc_src), (0, max_length - len(enc_src)), mode='constant', value=VOCAB['<PAD>']).unsqueeze(0).to(device)

    return enc_src, dec_src


def chat_with_transformer(model, question, max_length=50, temperature=1):
    model.eval()
    with torch.no_grad():
        enc_src, dec_src = prepare_model_input(question, max_length=max_length)

        # Placeholder for the generated answer
        generated_answer = []
        for i in range(max_length):
            # Forward pass through the model
            logits = model(enc_src, dec_src)

            # Get the token with the highest probability for the next position from the last time step
            predictions = F.softmax(logits / temperature, dim=2)[:, i, :]
            predicted_token = torch.multinomial(predictions, num_samples=1).squeeze(1)

            # Break if the EOS token is predicted
            if predicted_token.item() == VOCAB['<EOS>']:
                break

            # Append the predicted token to the decoder's input for the next time step
            dec_src = torch.cat((dec_src, predicted_token.unsqueeze(0)), dim=1)

            # Append the predicted token to the generated answer
            generated_answer.append(predicted_token.item())

        # Convert the generated tokens to words
        return tokens_to_text(generated_answer, VOCAB)
```

What we are doing is, we take a user input (question in our case) we pass the question to prepare_model_input function which will tokenize the question and will return decoder's input and a tensor with batch dimension which we will pass to model for the prediction one we get the predicted token we pass that to tokens_to_text to get the answer.

Finally the whole code all together would look like,

```python
import torch
import pickle
from chatbot.utils import get_device
from chatbot.data_load import text_to_tokens
from chatbot.data_load import tokens_to_text
import torch.nn.functional as F
from chatbot.model import TransformerPytroch

#Loading VOCAB
with open('/application/chatbot/vocab.pkl', 'rb') as f:
    VOCAB = pickle.load(f)
device = get_device()

def prepare_model_input(question, max_length=50):
    # Tokenize the input question
    tokenized_question = text_to_tokens(question, VOCAB)
    enc_src = tokenized_question + [VOCAB['<EOS>']]  # Add SOS and EOS tokens

    # Prepare a placeholder for the decoder's input
    dec_src = torch.LongTensor([VOCAB['<SOS>']]).unsqueeze(0).to(device)

    # Convert to tensor and add batch dimension
    enc_src = F.pad(torch.LongTensor(enc_src), (0, max_length - len(enc_src)), mode='constant', value=VOCAB['<PAD>']).unsqueeze(0).to(device)

    return enc_src, dec_src


def chat_with_transformer(model, question, max_length=50, temperature=1):
    model.eval()
    with torch.no_grad():
        enc_src, dec_src = prepare_model_input(question, max_length=max_length)

        # Placeholder for the generated answer
        generated_answer = []
        for i in range(max_length):
            # Forward pass through the model
            logits = model(enc_src, dec_src)

            # Get the token with the highest probability for the next position from the last time step
            predictions = F.softmax(logits / temperature, dim=2)[:, i, :]
            predicted_token = torch.multinomial(predictions, num_samples=1).squeeze(1)

            # Break if the EOS token is predicted
            if predicted_token.item() == VOCAB['<EOS>']:
                break

            # Append the predicted token to the decoder's input for the next time step
            dec_src = torch.cat((dec_src, predicted_token.unsqueeze(0)), dim=1)

            # Append the predicted token to the generated answer
            generated_answer.append(predicted_token.item())

        # Convert the generated tokens to words
        return tokens_to_text(generated_answer, VOCAB)
    
def init_chatbot():
    print("[~] Loading ChatBot")
    pytorch_load_model  = TransformerPytroch(
    inp_vocab_size = 958,
    trg_vocab_size = 958,
    src_pad_idx = 0,
    trg_pad_idx = 0,
    emb_size = 512,
    n_layers=1,
    heads=4,
    forward_expansion=4,
    drop_out=0.1,
    max_seq_len=10,
    device=device
    ).to(device)
    
    pytorch_load_model.load_state_dict(torch.load('/application/chatbot/pytorch_transformer_model.pth', map_location=torch.device('cpu')))
    return pytorch_load_model

    


def main_chatbot(model_loaded, question):
    transformer_response = chat_with_transformer(model_loaded, str(question), max_length=10, temperature=1.0)
    return transformer_response

```






