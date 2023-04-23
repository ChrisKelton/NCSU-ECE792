import re
from pathlib import Path
from typing import List, Optional

import nltk
import numpy as np
import torch.optim
from bs4 import BeautifulSoup
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import OneHotEncoder
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from lstm import LSTM
from mlp_mnist.training_utils import Loss, save_loss_plot


# Remove noise from text
def denoise_text(text):
    # remove html strips
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    # replace \n with space
    text = re.sub('\n', ' ', text)
    # remove square brackets
    text = re.sub('\[[^]]*\]', '', text)
    # replace punctuation with space
    text = re.sub(r'[,.;@#?!&$\-]+\*', ' ', text, flags=re.VERBOSE)
    # remove special characters
    text = re.sub(r'[^a-zA-z0-9\s]', '', text)
    # replace extra spaces with single space
    text = re.sub(' +', ' ', text)

    return text.lower().strip()


def get_denoised_gutenberg_austen_sense_txt():
    nltk.download('gutenberg')
    corpus = nltk.corpus.gutenberg.raw('austen-sense.txt')

    return denoise_text(corpus)


def get_batch(source, i, n_gram: int = 5, evaluation=False):
    seq_len = min(n_gram, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+n_gram:i+1+seq_len].view(-1))

    return data, target


def batch_sequences(data: torch.Tensor, batch_size: int):
    nbatch = data.size(0) // batch_size
    data = data.narrow(0, 0, nbatch * batch_size)
    # use contiguous to get transposed matrix into appropriate form in memory
    data = data.view(batch_size, -1).t().contiguous()

    return data


def remove_sequence_of_incorrect_length(sequences: List[List[int]], n: int) -> List[List[int]]:
    seq_to_remove = []
    for idx, seq in enumerate(sequences):
        if len(seq) != n:
            seq_to_remove.append(idx)

    print(f"Found '{len(seq_to_remove)}' sequences not of length '{n}'.")
    if len(seq_to_remove) > 0:
        seq_to_remove.sort(reverse=True)
        for seq_idx in seq_to_remove:
            sequences.pop(seq_idx)

    return sequences


def evaluate_test_case(model: LSTM, tokenizer: Tokenizer, device: Optional[torch.device] = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_eval_words = [["his", "natural", "shyness", "was", "overcome"]]
    eval_words = [original_eval_words[0].copy()]
    eval_sequence = tokenizer.texts_to_sequences(eval_words)
    eval_sequence_emb = batch_sequences(torch.tensor(eval_sequence), 1).t().contiguous()

    n_words_to_predict = 100
    predicted_words = []
    model.eval()
    h_t_1 = None
    C_t_1 = None
    for i in range(n_words_to_predict):
        output, h_t_1, C_t_1 = model(eval_sequence_emb, h_t_1, C_t_1)
        maxarg = (torch.argmax(output, dim=-1) + 1).view(1, 1).detach().cpu().numpy()
        predicted_word = tokenizer.sequences_to_texts(maxarg)[0]
        predicted_words.append(predicted_word)

        eval_words[0].pop(0)
        eval_words[0].append(predicted_word)
        eval_sequence = tokenizer.texts_to_sequences(eval_words)
        eval_sequence_emb = batch_sequences(torch.tensor(eval_sequence), 1).t().contiguous()

    full_text = original_eval_words[0].copy()
    full_text.extend(predicted_words)
    print(f"full text:\n{' '.join(full_text)}")


def main():
    base_out_path = Path("./")

    corpus = get_denoised_gutenberg_austen_sense_txt()

    words = corpus.split(' ')
    unique_words = np.unique(words)
    vocab_size = len(unique_words)

    print(f"vocab size: {vocab_size}")

    lr: float = 1e-3
    n: int = 6
    batch_size: int = 88
    n_epochs: int = 1000
    emb_size: int = 200
    n_hidden_neurons: int = 200
    alpha_ls: Optional[float] = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(words)
    n_words_per_sequence = [words[idx:n + idx] for idx in range(len(words) - n + 1)]
    # TODO: 'producing' which is the last element in tokenizer.word_index is not being included when performing texts_to_sequences; therefore, some sequences are n-1 length rather than all being uniform length. This only occurs near the end of all the text, so will ignore for now.
    sequences = tokenizer.texts_to_sequences(n_words_per_sequence)
    sequences = torch.tensor(remove_sequence_of_incorrect_length(sequences, n))
    sequences_emb = batch_sequences(sequences, batch_size)

    lstm_net = LSTM(vocab_size, emb_size, n_hidden_neurons)
    optimizer = torch.optim.Adam(lstm_net.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    enc = OneHotEncoder()
    enc.fit([[i] for i in range(1, vocab_size + 1)])

    losses = Loss.init()
    predicted_words = []
    h_t_1 = None
    C_t_1 = None
    for epoch in tqdm(range(n_epochs)):
        for batch, i in enumerate(range(0, sequences_emb.size(0), n)):
            x_t, target = get_batch(sequences_emb, i)
            target = torch.tensor(enc.transform(target.view(batch_size, 1)).toarray()).to(device)
            if alpha_ls is not None:
                # label smoothing to reduce penalty of incorrect guesses, since there are so many entries for the one
                # hot encoded labels
                target = (1 - alpha_ls) * target + alpha_ls / target.shape[-1]
            x_t = x_t.t().contiguous()
            optimizer.zero_grad()
            lstm_net.zero_grad()

            if h_t_1 is not None and C_t_1 is not None:
                h_t_1 = Variable(h_t_1).to(device)
                C_t_1 = Variable(C_t_1).to(device)
            output, h_t_1, C_t_1 = lstm_net(x_t, h_t_1, C_t_1)

            predicted_enc_out = output[:, -1, :]
            argmax = (torch.argmax(predicted_enc_out, dim=1) + 1).view(batch_size, 1).detach().cpu()
            predicted_words.append(tokenizer.sequences_to_texts((argmax.tolist())))

            loss = criterion(predicted_enc_out, target)
            loss.backward()
            optimizer.step()

            losses += loss.item()

            if len(predicted_words) % 50 == 0:
                print(f"[%d/%d]\tLoss: %.5f" % (len(predicted_words) / (epoch + 1), int(len(sequences) / batch_size), losses.current_loss))

        losses.update_for_epoch()
        scheduler.step(losses.loss_vals_per_epoch[-1])

    loss_base_path = base_out_path / "lstm" / "loss"
    loss_base_path.mkdir(exist_ok=True, parents=True)
    loss_path = loss_base_path / f"epoch--{n_epochs}.png"
    save_loss_plot(losses.loss_vals_per_epoch, loss_path)

    model_base_path = base_out_path / "lstm" / "models"
    model_path = model_base_path / f"LSTM--{n_epochs}.pth"
    torch.save(
        {
            f"LSTM-{n}-Gram": lstm_net.state_dict(),
        },
        str(model_path),
    )


if __name__ == '__main__':
    main()
