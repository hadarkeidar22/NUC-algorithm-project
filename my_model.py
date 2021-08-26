from torch import nn

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride = 1, padding= 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride = 1, padding= 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7, stride = 1, padding= 1),
            nn.ReLU())

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride = 1, padding= 1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride = 1, padding= 2, output_padding=0),
            nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

