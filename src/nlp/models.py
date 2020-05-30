from torch import nn

__all__ = ["BOWClassifier"]


class BOWClassifier(nn.Module):
    """A BoW classifier runs one-hot tokens through a dense layer with a bias."""

    def __init__(self, vocab_size: int, class_size: int, dropout: float = 0):
        super().__init__()
        self.bow = nn.Linear(in_features=vocab_size, out_features=class_size)
        nn.init.xavier_uniform_(self.bow.weight)  # Weight inits from sample
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input):
        return self.dropout(self.bow(input))
