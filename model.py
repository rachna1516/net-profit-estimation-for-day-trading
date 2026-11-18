import torch
import torch.nn as nn


class Gatedmodel(nn.Module):
    """Two stacked LSTM layers followed by a linear head.

    Input shape:  (batch, seq_length, input_size)
    Output shape: (batch, output_size)

    Note: ``nn.LSTM``'s own ``dropout`` argument only applies *between* layers
    of a multi-layer LSTM, so it has no effect with ``num_layers=1``. Dropout is
    therefore applied explicitly between the two LSTM layers and before the head.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size1: int,
        hidden_size2: int,
        output_size: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out1, _ = self.lstm1(x)
        lstm_out2, _ = self.lstm2(self.dropout1(lstm_out1))
        final_hidden_state = lstm_out2[:, -1, :]
        return self.fc(self.dropout2(final_hidden_state))
