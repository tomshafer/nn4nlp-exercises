from torch import nn

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

        self.dropout = nn.Dropout(p=dropout)

        self.softmax = nn.Linear(in_features=embedding_size, out_features=class_size)
        nn.init.xavier_uniform_(self.softmax.weight)  # Weight inits from sample

    def forward(self, input):
        y = self.embed(input).sum(dim=1)
        y = self.dropout(y)
        return self.softmax(y)
