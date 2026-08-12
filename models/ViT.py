import torch
from vit_pytorch import ViT

class FeatureExtractorViT(ViT):
    def __init__(self, image_size, channels, patch_size, dim, depth, heads, mlp_dim, pool='cls', dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=0,  # no classification head, set num_classes to 0
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            channels=channels,
            pool=pool,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout
        )
        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        # remove classification head
        del self.mlp_head

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :n]
        x = self.dropout(x)
        x = self.transformer(x)

        if self.pool == 'mean':
            x = x.mean(dim=1)
        else:
            x = x[:, :]

        h = self.image_size[0] // self.patch_size[0]
        w = self.image_size[1] // self.patch_size[1]
        # reshape to [B, N, H, W]
        x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2)

        return x

if __name__ == "__main__":
    # example parameters
    image_size = (960, 540)  # input image size
    channels = 4  # number of input channels
    patch_size = (80, 45)  # patch size
    dim = 512  # embedding dimension
    depth = 6  # number of transformer blocks
    heads = 16  # number of attention heads
    mlp_dim = 1024  # MLP hidden dimension

    model = FeatureExtractorViT(
        image_size=image_size,
        channels=channels,
        patch_size=patch_size,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim
    ).to('cuda')


    # test model (example) -- no debug printing
    # input shape [B, C, H, W] = [2, 4, 540, 960]
    x = torch.randn(2, 4, 540, 960).to('cuda')
    _ = model(x)