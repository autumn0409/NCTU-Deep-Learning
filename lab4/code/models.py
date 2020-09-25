import torch
import torch.nn as nn

from utils import SOS_token, MAX_LENGTH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, latent_size, cond_embedding_size):
        super(EncoderRNN, self).__init__()

        self.input_size = 28
        self.hidden_size = hidden_size
        self.cond_embedding_size = cond_embedding_size

        self.input_embedding = nn.Embedding(self.input_size, hidden_size)
        self.cond_embedding = nn.Embedding(4, cond_embedding_size)

        self.lstm = nn.LSTM(input_size=hidden_size,
                            hidden_size=hidden_size, batch_first=True)

    def forward(self, input, condition):
        batch_size = input.size(0)

        embedded_cond = self.cond_embedding(condition)
        h_0 = torch.zeros(batch_size, self.hidden_size -
                          self.cond_embedding_size).to(device)
        h_0 = torch.cat((h_0, embedded_cond), dim=1)
        h_0 = h_0.view(1, batch_size, self.hidden_size)
        c_0 = torch.zeros(1, batch_size, self.hidden_size).to(device)

        input = self.input_embedding(input)

        _, (h_n, _) = self.lstm(input, (h_0, c_0))
        return h_n


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size):
        super(DecoderRNN, self).__init__()

        self.output_size = 28
        self.hidden_size = hidden_size

        self.input_embedding = nn.Embedding(self.output_size, hidden_size)

        self.lstm = nn.LSTM(input_size=hidden_size,
                            hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, self.output_size)

    def forward(self, input, h_0, c_0):
        input = self.input_embedding(input)
        output, (h_n, c_n) = self.lstm(input, (h_0, c_0))
        output = self.fc(output)
        return output, h_n, c_n


class CVAE(nn.Module):
    def __init__(self, hidden_size, latent_size, cond_embedding_size):
        super(CVAE, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = 28

        self.encoder = EncoderRNN(hidden_size=hidden_size, latent_size=latent_size,
                                  cond_embedding_size=cond_embedding_size)
        self.decoder = DecoderRNN(hidden_size=hidden_size)

        self.cond_embedding = nn.Embedding(4, cond_embedding_size)
        self.latent_cond_embedding = nn.Linear(
            latent_size + cond_embedding_size, hidden_size)

        self.hidden2mean = nn.Linear(hidden_size, latent_size)
        self.hidden2logvar = nn.Linear(hidden_size, latent_size)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        latent = mean + eps * std
        return latent

    def latent_cond2hidden(self, latent, condition):
        batch_size = condition.size(0)

        embedded_cond = self.cond_embedding(condition)
        embedded_cond = embedded_cond.view(1, batch_size, -1)
        latent_cond = torch.cat((latent, embedded_cond), dim=2)
        hidden = self.latent_cond_embedding(latent_cond)

        return hidden

    def forward(self, input, condition, use_teacher_forcing):
        return self.__forward(input, condition, condition, use_teacher_forcing, 'train')

    def inference(self, input, input_condition, target_condition):
        outputs, _, _ = self.__forward(
            input, input_condition, target_condition, False, 'inference')
        return outputs

    def generation(self, latent, condition):
        batch_size = 1
        seq_length = MAX_LENGTH

        return self.__generation(latent, condition, batch_size, seq_length, False, None)

    def __forward(self, input, input_condition, target_condition, use_teacher_forcing, mode):
        batch_size = input.size(0)
        seq_length = MAX_LENGTH
        if mode == 'train':
            seq_length = input.size(1)

        hidden = self.encoder(input, input_condition)
        mean = self.hidden2mean(hidden)
        logvar = self.hidden2logvar(hidden)
        latent = self.reparameterize(mean, logvar)

        outputs = self.__generation(
            latent, target_condition, batch_size, seq_length, use_teacher_forcing, input)

        return outputs, mean, logvar

    def __generation(self, latent, target_condition, batch_size, seq_length, use_teacher_forcing, input):
        decoder_input = torch.tensor(
            [[SOS_token] * batch_size], device=device).view(batch_size, 1)

        decoder_hidden = self.latent_cond2hidden(latent, target_condition)
        decoder_cell = torch.zeros(1, batch_size, self.hidden_size).to(device)

        outputs = torch.zeros(batch_size, seq_length,
                              self.output_size).to(device)

        for di in range(seq_length):
            decoder_output, decoder_hidden, decoder_cell = self.decoder(
                decoder_input, decoder_hidden, decoder_cell)
            outputs[:, di, :] = decoder_output.view(
                batch_size, self.output_size)

            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                decoder_input = input[:, di].view(batch_size, 1)
            else:
                # Without teacher forcing: use its own predictions as the next input
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach().view(
                    batch_size, 1)  # detach from history as input

        return outputs
