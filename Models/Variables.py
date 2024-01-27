class Variables:
    def __init__(self, in_channels=3, embedding_size=256, batch_size=100, lr=0.001, margin=0.2, p=3, epochs=100):
        self.batch_size = batch_size
        self.lr = lr
        self.in_channels = in_channels
        self.emembedding_size =embedding_size
        self.p = p
        self.margin = margin
        self.epochs = epochs