import torch
import unittest
import numpy as np
import numpy.testing as npt


class TestTransformer(unittest.TestCase):

    def setUp(self):
        self.dim = 5
        self.emb = 3

        self.model = RobertaConstrastiveHead(self.dim, self.emb)
        self.model.u_embeddings.weight = torch.from_numpy(
            np.ones((self.emb, self.dim))
        )
        self.model.u_embeddings.bias = torch.from_numpy(
            np.ones(self.emb)
        )
        self.model.v_embeddings.weight = torch.from_numpy(
            np.ones((self.dim, self.emb))
        )
        self.model.v_embeddings.bias = torch.from_numpy(
            np.ones(self.dim)
        )

    def test_forward(self):
        inp1 = torch.ones(self.dim)
        inp2 = torch.ones(self.dim)
        inp3 = torch.cat(
            -1 * torch.ones(self.dim),
            -1.5 * torch.ones(self.dim),
            -2 * torch.ones(self.dim)
        )
        out = self.model.forward(inp1, inp2, inp3)
        res = out.detach().numpy()
        exp = np.array([19] * 5)


if __name__ == '__main__':
    unittest.main()
