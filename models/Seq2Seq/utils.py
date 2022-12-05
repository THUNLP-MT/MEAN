class Seq2SeqConfig:
    def __init__(self, vocab_size, cdr_type='3', hidden_size=256,
                 depth=2, dropout=0.1):
        self.vocab_size = vocab_size
        self.cdr_type = cdr_type
        self.hidden_size = hidden_size
        self.depth = depth
        self.dropout = dropout