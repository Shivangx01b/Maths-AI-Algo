import torch
import pickle
from aimodel import utils
from aimodel import data_load
import torch.nn.functional as F
from aimodel import model

VOCAB_SIZE = 958
#Loading VOCAB
with open('aimodel\\vocab_math_4.pkl', 'rb') as f:
    VOCAB = pickle.load(f)
device = utils.get_device()

def prepare_model_input(question, max_length=50):
    # Tokenize the input question
    tokenized_question = data_load.text_to_tokens(question, VOCAB)
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
        return data_load.tokens_to_text(generated_answer, VOCAB)
    
def init_chatbot():
    print("[~] Loading ChatBot")
    pytorch_transformer = model.TransformerPytroch(
    inp_vocab_size=VOCAB_SIZE,
    trg_vocab_size=VOCAB_SIZE,
    src_pad_idx=0,
    trg_pad_idx=0,
    emb_size=512,
    n_layers=1,
    heads=4,
    forward_expansion=4,
    drop_out=0.1,
    max_seq_len=10,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
    state_dict = torch.load('aimodel\\math_model4.pth', map_location='cuda' if torch.cuda.is_available() else 'cpu')
    pytorch_transformer.load_state_dict(state_dict)
    return  pytorch_transformer.to(device)

    


def main_chatbot(model_loaded, question):
    transformer_response = chat_with_transformer(model_loaded, str(question), max_length=10, temperature=1.0)
    return transformer_response

