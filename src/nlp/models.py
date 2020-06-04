from torch import nn
from torch.nn import functional as F

__all__ = ["BOWClassifier", "CBOWClassifier"]


class BOWClassifier(nn.Module):
    """A BoW classifier runs one-hot tokens through a dense layer with a bias."""

    def __init__(self, vocab_size: int, class_size: int, dropout: float = 0):
        super().__init__()
        self.bow = nn.Linear(in_features=vocab_size, out_features=class_size)
        nn.init.xavier_uniform_(self.bow.weight)  # Weight inits from sample
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input):
        return self.dropout(self.bow(input))


class CBOWClassifier(nn.Module):
    """A CBoW classifier runs embeddings through a dense layer with a bias."""

    def __init__(
        self,
        vocab_size: int,
        class_size: int,
        embedding_size: int = 64,
        dropout: float = 0,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=0,)
        nn.init.xavier_uniform_(self.embed.weight)

        self.softmax = nn.Linear(in_features=embedding_size, out_features=class_size)
        nn.init.xavier_uniform_(self.softmax.weight)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, input):
        y = self.embed(input).sum(dim=1)
        return self.softmax(self.dropout2(y))


class DeepCBOWClassifier(nn.Module):
    """A "deep" CBoW classifier adds a nonlinearity to the CBoW classifier."""

    def __init__(
        self,
        vocab_size: int,
        class_size: int,
        embedding_size: int = 64,
        hidden_size: int = 64,
        dropout: float = 0,
    ):
        super().__init__()

        # Embeddings
        self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=0,)
        nn.init.xavier_uniform_(self.embed.weight)
        # self.dropout_embed = nn.Dropout(p=dropout)

        # Hidden layer; cf sample code
        self.hidden = nn.ModuleList(
            [
                nn.Linear(in_features=embedding_size, out_features=hidden_size),
                nn.Dropout(p=dropout),
                nn.ReLU(),
                nn.Linear(in_features=hidden_size, out_features=hidden_size),
                nn.Dropout(p=dropout),
                nn.ReLU(),
            ]
        )
        nn.init.xavier_uniform_(self.hidden[0].weight)
        nn.init.xavier_uniform_(self.hidden[3].weight)

        # Softmax
        self.softmax = nn.Linear(in_features=hidden_size, out_features=class_size)
        nn.init.xavier_uniform_(self.softmax.weight)
        self.dropout_softmax = nn.Dropout(p=dropout)

    def forward(self, x):
        y = self.embed(x).sum(dim=1)
        for layer in self.hidden:
            y = layer(y)
        y = self.dropout_softmax(self.softmax(y))
        return y
