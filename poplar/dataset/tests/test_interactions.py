import unittest
import numpy as np
from poplar.util import get_data_path
import pandas as pd
from Bio import SeqIO
from poplar.dataset.interactions import (
    InteractionDataset, parse, preprocess,
    clean, dictionary, NegativeSampler)


class TestPreprocess(unittest.TestCase):

    def setUp(self):
        self.links_file = get_data_path('links.txt')
        self.fasta_file = get_data_path('prots.fa')

    def test_preprocess(self):
        seqs = list(SeqIO.parse(self.fasta_file, format='fasta'))
        links = pd.read_table(self.links_file, header=None)
        truncseqs = list(map(clean, seqs))
        seqids = list(map(lambda x: x.id, truncseqs))
        seqdict = dict(zip(seqids, truncseqs))
        pairs = preprocess(seqdict, links)
        self.assertListEqual(list(pairs.shape), [119, 2])


class TestInteractionDataset(unittest.TestCase):

    def setUp(self):
        self.links_file = get_data_path('links.txt')
        self.fasta_file = get_data_path('prots.fa')

        self.seqs = list(SeqIO.parse(self.fasta_file, format='fasta'))
        links = pd.read_table(self.links_file, header=None, index_col=0)

        truncseqs = list(map(clean, self.seqs))
        seqids = list(map(lambda x: x.id, truncseqs))
        seqdict = dict(zip(seqids, truncseqs))
        self.pairs = preprocess(seqdict, links)

    def test_random_peptide_draw(self):
        np.random.seed(0)
        intsd = InteractionDataset(self.pairs)
        seq = intsd.random_peptide_draw()
        self.assertEqual(len(seq), 200)

    def test_random_peptide_uniform(self):
        np.random.seed(0)
        intsd = InteractionDataset(self.pairs)
        seq = intsd.random_peptide_uniform()
        self.assertGreater(len(seq), 30)
        self.assertLess(len(seq), 1024)
        self.assertTrue(
            set(seq).issubset(set(dictionary.keys()))
        )
        self.assertNotIn('.', seq)

    def test_random_peptide(self):
        # Test the random_peptide function
        # to make sure that peptides are sampled
        # uniformly from the database
        np.random.seed(0)
        sampler = NegativeSampler(self.seqs)
        intsd = InteractionDataset(self.pairs, sampler)
        res = intsd.random_peptide()
        seqset = list(map(clean, self.seqs))
        seqset = set(map(lambda x: x.seq, seqset))
        self.assertIn(res, seqset)

    def test_getitem(self):
        np.random.seed(0)
        sampler = NegativeSampler(self.seqs)
        intsd = InteractionDataset(self.pairs, sampler)
        gene, pos, neg = intsd[0]

        exp_gene = list(
            'MINEIKKEAQERMGKTLEALGHAFAKIRTGRAHPSILDSVMVSYYGADTPLRQVANVTV'
            'EDSRTLALAVFDKSMIQAVEKAIMTSDLGLNPATAGTTIRVPMPALTEETRKGYTKQAR'
            'AEAEQARVSVRNIRRDALAQLKDLQKEKEISEDEERRAGDDVQKLTDKFIGEIEKALEA'
            'KEADLMAV'
        )
        exp_pos = list(
            'MMRSHYCGQLNESLDGQEVTLCGWVHRRRDHGGVIFLDVRDREGLAQVVFDPDRAETFA'
            'KADRVRSEFVVKITGKVRLRPEGARNPNMASGSIEVLGYELEVLNQAETPPFPLDEYSD'
            'VGEETRLRYRFIDLRRPEMAAKLKLRARITSSIRRYLDDNGFLDVETPILGRPTPEGAR'
            'DYLVPSRTYPGHFFALPQSPQLFKQLLMVAGFDRYYQIAKCFRDEDLRADRQPEFTQID'
            'IETSFLDESDIIGITEKMVRQLFKEVLDVEFDEFPHMPFEEAMRRYGSDKPDLRIPLEL'
            'VDVADQLKEVEFKVFSGPANDPKGRVAALRVPGAASMPRSQIDDYTKFVGIYGAKGLAY'
            'IKVNERAKGVEGLQSPIVKFIPEANLNVILDRVGAVDGDIVFFGADKAKIVCDALGALR'
            'IKVGHDLKLLTREWAPMWVVDFPMFEENDDGSLSALHHPFTSPKCTPAELEANPGAALS'
            'RAYDMVLNGTELGGGSIRIHDKSMQQAVFRVLGIDEAEQEEKFGFLLDALKYGAPPHGG'
            'LAFGLDRLVMLMTGASSIREVIAFPKTQSAGDVMTQAPGSVDGKALRELHIRLREQPKAE'
        )
        exp_neg = list(
            'MTTSDLPAFWTVIPAAGVGSRMRADRPKQYLDLAGRTVIERTLDCFLEHPMLRGLVVCL'
            'AEDDPYWPGLDCAASRHVQRAAGGVERADSVLSGLLRLLELGAQADDWVLVHDAARPNL'
            'TRGDLDRLLEELAEDPVGGLLAVPARDTLKRSDRDGRVSETIDRSVVWLAYTPQMFRLG'
            'ALHRALADALVAGVAITDEASAMEWAGYAPKLVEGRADNLKITTPEDLLRLQRSFPH'
        )

        self.assertListEqual(list(gene), exp_gene)
        self.assertListEqual(list(pos), exp_pos)
        self.assertListEqual(list(neg), exp_neg)

    def test_getitem_truncate(self):
        np.random.seed(0)
        intsd = InteractionDataset(self.pairs)
        gene, pos, neg = intsd[98]
        self.assertEqual(len(gene), 1024)

    def test_iter(self):
        # Test the iter function to make sure
        # negative samples are being drawn
        np.random.seed(0)
        sampler = NegativeSampler(self.seqs)
        intsd = InteractionDataset(self.pairs, sampler)
        res = [r for r in intsd]
        self.assertEqual(len(res), self.pairs.shape[0] * intsd.num_neg)


    def test_parse(self):
        # TODO: make sure that a validate dataloader is added
        res = parse(self.fasta_file, self.links_file,
                    training_column='Training',
                    batch_size=10, num_workers=1, arm_the_gpu=False)
        train, test = res
        for g, p, n in train:
            assert True
        self.assertEqual(len(res), 2)

    def test_validator(self):
        # TODO: make sure that a validate dataloader is added
        # this will take in some negative samples from
        # http://www.russelllab.org/negatives/
        pass


if __name__ == "__main__":
    unittest.main()
