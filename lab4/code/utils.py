import torch
import torch.nn as nn
from torch import optim

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from tqdm import tqdm
# import numpy as np
import random
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# simple present(sp), third person(tp), present progressive(pg), simple past(p).
TENSES = ['sp', 'tp', 'pg', 'p']
CHARS = ['SOS', 'EOS'] + [chr(i) for i in range(ord('a'), ord('z')+1)]

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 20
output_size = 28


def compute_bleu(output, reference):
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33, 0.33, 0.33)
    else:
        weights = (0.25, 0.25, 0.25, 0.25)
    return sentence_bleu([reference], output, weights=weights, smoothing_function=cc.method1)


def Gaussian_score(words):
    words_list = []
    score = 0
    yourpath = './train.txt'  # should be your directory of train.txt
    with open(yourpath, 'r') as fp:
        for line in fp:
            word = line.split(' ')
            word[3] = word[3].strip('\n')
            words_list.extend([word])
        for t in words:
            for i in words_list:
                if t == i:
                    score += 1
    return score/len(words)


def word2tensor(word):
    indexes = [CHARS.index(char) for char in word]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long)


def tense2tensor(tense):
    index = TENSES.index(tense)
    return torch.tensor([index], dtype=torch.long)


def outputs2word(outputs):
    predictions = []
    for i in range(outputs.size(1)):
        output = outputs[0][i].data.cpu()
        topv, topi = output.topk(1)
        prediction = topi.item()
        if prediction == EOS_token:
            break
        predictions.append(CHARS[prediction])

    return ''.join(predictions)


def word_gen(cvae, z, tense):
    condition = tense2tensor(tense).to(device)
    z = z.to(device)

    with torch.no_grad():
        outputs = cvae.generation(z, condition)

    return outputs2word(outputs)


def word_pred(cvae, word, input_tense, target_tense):
    batch_size = 1

    input = word2tensor(word).view(batch_size, -1).to(device)
    input_condition = tense2tensor(input_tense).to(device)
    target_condition = tense2tensor(target_tense).to(device)

    with torch.no_grad():
        outputs = cvae.inference(input, input_condition, target_condition)

    return outputs2word(outputs)


def eval_model_gaussian(cvae, verbose=False):
    latent_size = 32
    generate_words = []

    for i in range(100):
        z = torch.randn(1, 1, latent_size)
        generate_tenses = []

        for tense in TENSES:
            w = word_gen(cvae, z, tense=tense)
            generate_tenses.append(w)

        if verbose:
            print(generate_tenses)

        generate_words.append(generate_tenses)

    gaussian_score = Gaussian_score(generate_words)
    if verbose:
        print(f'Gaussian score: {gaussian_score}')

    return gaussian_score


def eval_model_bleu(cvae, verbose=False):
    input_tenses = ['sp', 'sp', 'sp', 'sp', 'p', 'sp', 'p', 'pg', 'pg', 'pg']
    target_tenses = ['p', 'pg', 'tp', 'tp', 'tp', 'pg', 'sp', 'sp', 'p', 'tp']
    inputs = []
    targets = []

    with open('test.txt', 'r') as f:
        for line in f:
            data = line.split('\n')[0].split(' ')
            inputs.append(data[0])
            targets.append(data[1])

    score_sum = 0
    for i in range(len(inputs)):
        prediction = word_pred(
            cvae, word=inputs[i], input_tense=input_tenses[i], target_tense=target_tenses[i])
        score = compute_bleu(prediction, targets[i])
        if verbose:
            print(f'input: {inputs[i]}')
            print(f'target: {targets[i]}')
            print(f'prediction: {prediction}')
            print()
        score_sum += score

    if verbose:
        print(f'Average BLEU-4 score: {score_sum / 10}')

    return score_sum / 10


def KL_weight_schedule(epoch, n_epochs, KL_annealing_method):
    period = n_epochs // 3

    if KL_annealing_method == 'cyclical':
        epoch %= period

    KL_weight = epoch / period
    KL_weight = min(1, KL_weight)

    return KL_weight


def teacher_forcing_ratio_schedule(epoch, n_epochs):
    epoch -= 1
    teacher_forcing_ratio = 1 - (epoch / n_epochs)
    return teacher_forcing_ratio


# Loss Function
def loss_func(output, target, mean, logvar):
    # Cross Entropy Loss
    loss_func_ce = nn.CrossEntropyLoss()
    output = output.view(-1, output_size)
    target = target.view(-1)
    CE_loss = loss_func_ce(output, target)

    # KL Divergence
    KL_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

    return CE_loss, KL_loss


def trainEpochs(train_loader, cvae, n_epochs, learning_rate, KL_annealing_method):

    history = {"CE_loss": [], "KL_loss": [],
               "BLEU_score": [], "KL_weight": [], "TF_ratio": []}

    cvae_optimizer = optim.Adam(cvae.parameters(), lr=learning_rate)
    pbar = tqdm(total=n_epochs, unit=' epochs', ascii=True)

    for epoch in range(1, n_epochs + 1):

        total_KL_loss = total_CE_loss = 0
        KL_weight = KL_weight_schedule(epoch, n_epochs, KL_annealing_method)
        teacher_forcing_ratio = teacher_forcing_ratio_schedule(epoch, n_epochs)

        cvae.train()

        for batch_idx, (input, condition) in enumerate(train_loader):
            input, condition = input.to(device), condition.to(device)
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

            cvae_optimizer.zero_grad()
            output, mean, logvar = cvae(input, condition, use_teacher_forcing)

            CE_loss, KL_loss = loss_func(output, input, mean, logvar)
            loss = CE_loss + KL_weight * KL_loss

            loss.backward()
            cvae_optimizer.step()

            total_CE_loss += CE_loss.item()
            total_KL_loss += KL_loss.item()

        cvae.eval()

        # Record Epoch Loss
        history['CE_loss'].append(total_CE_loss / len(train_loader))
        history['KL_loss'].append(total_KL_loss / len(train_loader))
        history['BLEU_score'].append(eval_model_bleu(cvae))

        history['KL_weight'].append(KL_weight)
        history['TF_ratio'].append(teacher_forcing_ratio)

        if history['BLEU_score'][-1] > 0.8:
            torch.save(cvae, f'./model_weights/{KL_annealing_method}_epoch{epoch}.pkl')

        pbar.set_postfix({'CE_loss': history['CE_loss'][-1],
                          'KL_loss': history['KL_loss'][-1],
                          'BLEU_score': history['BLEU_score'][-1]})
        pbar.update()

    pbar.close()
    return history


def plot_loss(history, title):
    plt.plot(history['CE_loss'], label='CE Loss')
    plt.plot(history['KL_loss'], label='KL Loss')
    plt.legend(loc='best')
    plt.savefig(f'./graphs/{title}_loss')
    plt.close()


def plot_score(history, title):
    plt.plot(history['BLEU_score'], label='BLEU score')
    plt.legend(loc='best')
    plt.savefig(f'./graphs/{title}_score')
    plt.close()


def plot_ratio(history, title):
    plt.plot(history['KL_weight'], label='KL weight')
    plt.plot(history['TF_ratio'], label='TF ratio')
    plt.legend(loc='best')
    plt.savefig(f'./graphs/{title}_ratio')
    plt.close()


def plot_results(history, title):
    plot_ratio(history, title)
    plot_score(history, title)
    plot_loss(history, title)
