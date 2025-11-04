import torch
import torch.nn as nn

class MLPTextGen(nn.Module):
    def __init__(self, seq_len, vocab_size, embed_dim, hidden_dim, num_layers, activation='relu', dropout=0.5):
        super().__init__()
        self.seq_len = seq_len
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        layers = [nn.Linear(embed_dim * seq_len, hidden_dim)]
        act = nn.ReLU() if activation == 'relu' else nn.Tanh()
        layers.append(act)
        layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act)
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, vocab_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        embedded = self.embedding(x)                 # [batch, seq_len, embed_dim]
        flattened = embedded.view(embedded.size(0), -1)
        return self.layers(flattened)