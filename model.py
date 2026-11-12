import torch.nn as nn

class Gatedmodel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(Gatedmodel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True, num_layers=1, dropout=0.2)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True, num_layers=1, dropout=0.2)
        self.fc    = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        lstm_out1, _       = self.lstm1(x)
        lstm_out2, _       = self.lstm2(lstm_out1)
        final_hidden_state = lstm_out2[:, -1, :]
        output             = self.fc(final_hidden_state)
        return output