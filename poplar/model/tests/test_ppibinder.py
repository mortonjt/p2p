import torch
import unittest
import numpy as np
import numpy.testing as npt
from poplar.model.ppibinder import PPIBinder
from poplar.model.dummy import DummyModel
from poplar.util import dictionary, encode


class TestPPIBinder(unittest.TestCase):

    def setUp(self):
        self.input_size = len(dictionary)
        self.dim = 5
        self.emb = 3

        peptide_model = DummyModel(self.input_size, self.dim)
        peptide_model.encoder.weight = torch.nn.Parameter(
            torch.from_numpy(
                np.ones((self.input_size, self.dim), dtype=np.float32)
            )
        )

        self.model = PPIBinder(self.dim, self.emb, peptide_model)
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

        inp1 = torch.stack(list(map(encode, [
            'RAYDMVLNGTELGGGSIRIHDKSMQQAVFRVLGIDEAEQEEKFGFLLDALKYGAPPHGG'
        ])))
        inp2 = torch.stack(list(map(encode, [
            'RAYDMVLNGTELGGGSIRIHDKSMQQAVFRVLGIDEAEQEEKFGFLLDALKYGAPPHGG'
        ])))
        inp3 = torch.stack(list(map(encode, [
            'VDVADQLKEVEFKVFSGPANDPKGRVAALRVPGAASMPRSQIDDYTKFVGIYGAKGLAY'
        ])))

        # inp1 = torch.ones(1, self.dim) * 0.001
        # inp2 = torch.ones(1, self.dim) * 0.001
        # inp3 = -0.001 * torch.ones(1, self.dim)

        out = self.model.forward(inp1, inp2, inp3)
        res = out.detach().numpy()
        exp = np.array(108, dtype=np.float32)
        self.assertEqual(res, exp)

    def test_forward_batch(self):
        inp1 = torch.stack(list(map(encode, [
            'IKVNERAKGVEGLQSPIVKFIPEANLNVILDRVGAVDGDIVFFGADKAKIVCDALGALR',
            'IKVGHDLKLLTREWAPMWVVDFPMFEENDDGSLSALHHPFTSPKCTPAELEANPGAALS',
            'RAYDMVLNGTELGGGSIRIHDKSMQQAVFRVLGIDEAEQEEKFGFLLDALKYGAPPHGG'
        ])))
        inp2 = torch.stack(list(map(encode, [
            'IKVNERAKGVEGLQSPIVKFIPEANLNVILDRVGAVDGDIVFFGADKAKIVCDALGALR',
            'IKVGHDLKLLTREWAPMWVVDFPMFEENDDGSLSALHHPFTSPKCTPAELEANPGAALS',
            'RAYDMVLNGTELGGGSIRIHDKSMQQAVFRVLGIDEAEQEEKFGFLLDALKYGAPPHGG'
        ])))
        inp3 = torch.stack(list(map(encode, [
            'DYLVPSRTYPGHFFALPQSPQLFKQLLMVAGFDRYYQIAKCFRDEDLRADRQPEFTQID',
            'IETSFLDESDIIGITEKMVRQLFKEVLDVEFDEFPHMPFEEAMRRYGSDKPDLRIPLEL',
            'VDVADQLKEVEFKVFSGPANDPKGRVAALRVPGAASMPRSQIDDYTKFVGIYGAKGLAY'
        ])))
        batch = 3

        # inp1 = torch.ones(batch, self.dim) * 0.001
        # inp2 = torch.ones(batch, self.dim) * 0.001
        # inp3 = -0.001 * torch.ones(batch, self.dim)

        out = self.model.forward(inp1, inp2, inp3)
        res = out.detach().numpy()
        exp = np.array(108 * batch, dtype=np.float32)
        self.assertEqual(res, exp)

    def test_forward_batch_neg(self):
        # TODO: make sure that batch forward pass with
        # multiple negative samples works
        # also make sure that expectations are also
        # being correctly implemented (as shown in Levy 2014)
        batch = 3
        num_neg = 3
        inp1 = torch.stack(list(map(encode, [
            'IKVNERAKGVEGLQSPIVKFIPEANLNVILDRVGAVDGDIVFFGADKAKIVCDALGALR',
            'IKVGHDLKLLTREWAPMWVVDFPMFEENDDGSLSALHHPFTSPKCTPAELEANPGAALS',
            'RAYDMVLNGTELGGGSIRIHDKSMQQAVFRVLGIDEAEQEEKFGFLLDALKYGAPPHGG'
        ] * num_neg)))
        inp2 = torch.stack(list(map(encode, [
            'IKVNERAKGVEGLQSPIVKFIPEANLNVILDRVGAVDGDIVFFGADKAKIVCDALGALR',
            'IKVGHDLKLLTREWAPMWVVDFPMFEENDDGSLSALHHPFTSPKCTPAELEANPGAALS',
            'RAYDMVLNGTELGGGSIRIHDKSMQQAVFRVLGIDEAEQEEKFGFLLDALKYGAPPHGG'
        ] * num_neg)))
        inp3 = torch.stack(list(map(encode, [
            'DYLVPSRTYPGHFFALPQSPQLFKQLLMVAGFDRYYQIAKCFRDEDLRADRQPEFTQID',
            'IETSFLDESDIIGITEKMVRQLFKEVLDVEFDEFPHMPFEEAMRRYGSDKPDLRIPLEL',
            'VDVADQLKEVEFKVFSGPANDPKGRVAALRVPGAASMPRSQIDDYTKFVGIYGAKGLAY'
        ] * num_neg)))

        # inp1 = torch.ones(batch * num_neg, self.dim) * 0.001
        # inp2 = torch.ones(batch * num_neg, self.dim) * 0.001
        # inp3 = -0.001 * torch.ones(batch * num_neg, self.dim)

        out = self.model.forward(inp1, inp2, inp3)
        res = out.detach().numpy()
        exp = np.array(108 * num_neg * batch, dtype=np.float32)
        self.assertAlmostEqual(res, exp, places=4)


if __name__ == '__main__':
    unittest.main()
