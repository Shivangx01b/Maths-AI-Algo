
from data_load import main_fire_all_dataset
from utils import get_device
from model import TransformerPytroch
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