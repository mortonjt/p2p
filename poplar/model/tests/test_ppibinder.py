import torch
import unittest
import numpy as np
import numpy.testing as npt
from poplar.model.ppibinder import PPBinder


class TestPPBinder(unittest.TestCase):

    def setUp(self):
        self.dim = 5
        self.emb = 3

        self.model = PPBinder(self.dim, self.emb)

        self.model.u_embeddings.weight = torch.nn.Parameter(
            torch.from_numpy(
                np.ones((self.emb, self.dim), dtype=np.float32)
            )
        )
        self.model.u_embeddings.bias = torch.nn.Parameter(
            torch.from_numpy(
                np.ones(self.emb, dtype=np.float32)
            )
        )
        self.model.v_embeddings.weight = torch.nn.Parameter(
            torch.from_numpy(
                np.ones((self.emb, self.dim), dtype=np.float32)
            )
        )
        self.model.v_embeddings.bias = torch.nn.Parameter(
            torch.from_numpy(
                np.ones(self.emb, dtype=np.float32)
            )
        )

    def test_forward_single(self):
        inp1 = torch.ones(1, self.dim) * 0.001
        inp2 = torch.ones(1, self.dim) * 0.001
        inp3 = -0.001 * torch.ones(1, self.dim)

        out = self.model.forward(inp1, inp2, inp3)
        res = out.detach().numpy()
        exp = np.array(3.0956974, dtype=np.float32)
        self.assertEqual(res, exp)

    def test_forward_batch(self):
        batch = 5

        inp1 = torch.ones(batch, self.dim) * 0.001
        inp2 = torch.ones(batch, self.dim) * 0.001
        inp3 = -0.001 * torch.ones(batch, self.dim)

        out = self.model.forward(inp1, inp2, inp3)
        res = out.detach().numpy()
        exp = np.array(3.0956974 * batch, dtype=np.float32)
        self.assertEqual(res, exp)

    def test_forward_batch_neg(self):
        # TODO: make sure that batch forward pass with
        # multiple negative samples works
        # also make sure that expectations are also
        # being correctly implemented (as shown in Levy 2014)
        batch = 5
        num_neg = 3

        inp1 = torch.ones(batch * num_neg, self.dim) * 0.001
        inp2 = torch.ones(batch * num_neg, self.dim) * 0.001
        inp3 = -0.001 * torch.ones(batch * num_neg, self.dim)

        out = self.model.forward(inp1, inp2, inp3)
        res = out.detach().numpy()
        exp = np.array(3.0956974 * batch * num_neg, dtype=np.float32)
        self.assertAlmostEqual(res, exp, places=4)


if __name__ == '__main__':
    unittest.main()
