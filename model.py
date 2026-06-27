# many to many architecture
import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_layer_size=500, num_layers=2,output_size=6):
        super().__init__()

        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers

        # RNN layer
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_layer_size,
            num_layers=num_layers,
            batch_first=True

        )
        self.dropout = nn.Dropout(0.3)

        # Fully Connected Layer
        self.fc = nn.Linear(hidden_layer_size,output_size)

    def forward(self, x):
        # Initial hidden state
        h0 = torch.zeros(
            self.num_layers,
            x.size(0),
            self.hidden_layer_size
        ).to(x.device)

        # Forward pass through RNN
        out, _ = self.rnn(x, h0)

        # Take only the last timestep output
        out = out[:, -1, :]
        
        # dropout
        out = self.dropout(out)    #works only in training and automatically turns off in eval mode

        # Pass through fully connected layer
        out = self.fc(out)

        return out    