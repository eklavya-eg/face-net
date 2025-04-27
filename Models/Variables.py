class Variables:
    def __init__(self, in_channels=3, embedding_size=128, batch_size=4, batch_size_test_val = 4, lr=0.001, margin=0.5, p=2, epochs=100, step_size=3, gamma=0.1, p_dropout=0.2, p_linear_dropout=0.2):
        self.batch_size = batch_size
        self.lr = lr
        self.in_channels = in_channels
        self.emembedding_size =embedding_size
        self.p = p
        self.margin = margin
        self.epochs = epochs
        self.step_size = step_size
        self.gamma = gamma
        self.batch_size_test_val = batch_size_test_val
        self.p_dropout = p_dropout
        self.p_linear_dropout = p_linear_dropout