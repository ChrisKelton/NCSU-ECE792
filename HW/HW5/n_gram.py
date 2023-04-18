import nltk
import torch.optim
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
import numpy as np
from lstm import LSTM
from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from torch.autograd import Variable
from sklearn.preprocessing import OneHotEncoder
from mlp_mnist.training_utils import Loss, Accuracy, save_loss_plot
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path


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


def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


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
    n_epochs: int = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # reduced_vocab_size = np.power(vocab_size, 0.25)
    # reduced_vocab_size = np.max((2000, vocab_size // 4))
    # print(f"reduced vocab size: {reduced_vocab_size}")
    reduced_vocab_size = vocab_size
    tokenizer = Tokenizer(num_words=reduced_vocab_size)
    tokenizer.fit_on_texts(words)
    n_words_per_sequence = [words[idx:n + idx] for idx in range(len(words) - n + 1)]
    # TODO: 'producing' which is the last element in tokenizer.word_index is not being included when performing texts_to_sequences; therefore, some sequences are n-1 length rather than all being uniform length. This only occurs near the end of all the text, so will ignore for now.
    sequences = tokenizer.texts_to_sequences(n_words_per_sequence)

    lstm_net = LSTM(in_feats=n-1, out_feats=vocab_size, batch_size=batch_size).to(device)
    optimizer = torch.optim.Adam(lstm_net.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    enc = OneHotEncoder()
    # enc.fit([[i] for i in range(1, vocab_size + 1)])
    enc.fit([[i] for i in range(1, reduced_vocab_size + 1)])

    losses = Loss.init()
    predicted_words = []
    h_t_1 = lstm_net.init_hidden(batch_size).to(device)
    C_t_1 = lstm_net.init_hidden(batch_size).to(device)
    for epoch in tqdm(range(n_epochs)):
        sequence_to_learn = []
        target = []
        for sequence_idx, sequence in enumerate(sequences):
            if len(sequence) == n:
                sequence_to_learn.append(sequence[:-1])
                target.append(sequence[-1])
            if len(sequence_to_learn) == batch_size:
                sequence_to_learn = torch.tensor(sequence_to_learn).view(batch_size, n-1).to(device)
                # target = torch.tensor(target)
                target = torch.tensor(enc.transform(torch.tensor(target).view(batch_size, 1)).toarray()).to(device)
                optimizer.zero_grad()
                lstm_net.zero_grad()

                # X, y = sequence[:-1], sequence[-1]
                # X = torch.tensor(X)
                # if X.shape[0] != batch_size:
                #     X = X.view(1, len(X))
                # y = torch.tensor(y)
                h_t_1 = Variable(h_t_1).to(device)
                C_t_1 = Variable(C_t_1).to(device)
                h_t_1, C_t_1 = lstm_net(sequence_to_learn, h_t_1, C_t_1)
                # predicted_words.append(tokenizer.sequences_to_texts([[h_t_1.item()]]))
                # predicted_words.append(tokenizer.sequences_to_texts(np.round(h_t_1.detach().cpu().numpy())))
                # predicted_words.append(tokenizer.sequences_to_texts(np.round(C_t_1.detach().cpu().numpy())))
                argmax = (torch.argmax(torch.nn.functional.softmax(h_t_1, dim=1), dim=1) + 1).view(batch_size, 1).detach().cpu()
                predicted_words.append(tokenizer.sequences_to_texts((argmax.tolist())))

                # loss = criterion(h_t_1, y.to(torch.float32).view(1, 1))
                # loss = criterion(h_t_1, target.to(torch.float32).view(target.shape[0], 1))
                loss = criterion(h_t_1, target)
                # loss = criterion(C_t_1, target.to(torch.float32).view(target.shape[0], 1))
                loss.backward()
                optimizer.step()

                losses += loss.item()

                sequence_to_learn = []
                target = []
                if len(predicted_words) % 50 == 0:
                    print(f"[%d/%d]\tLoss: %.5f" % (len(predicted_words) / (epoch + 1), int(len(sequences) / batch_size), losses.current_loss))

        losses.update_for_epoch()
        scheduler.step(losses.loss_vals_per_epoch[-1])
        a = 0

    loss_base_path = base_out_path / "LSTM" / "models" / "loss"
    loss_base_path.mkdir(exist_ok=True, parents=True)
    loss_path = loss_base_path / f"epoch--{epoch+1}.png"
    save_loss_plot(losses.loss_vals_per_epoch, loss_path)

    model_base_path = base_out_path / "LSTM" / "models"
    model_path = model_base_path / f"LSTM--{epoch+1}.pth"
    torch.save(
        {
            f"LSTM-{n}-Gram": lstm_net.state_dict(),
        },
        str(model_path),
    )


if __name__ == '__main__':
    main()
