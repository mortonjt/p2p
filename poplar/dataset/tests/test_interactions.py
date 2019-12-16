import unittest
import numpy as np
from poplar.util import get_data_path
import pandas as pd
from Bio import SeqIO
from poplar.dataset.interactions import (
    InteractionDataset, ValidationDataset,
    parse, preprocess,
    clean, dictionary,
    NegativeSampler)


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
        self.assertListEqual(list(pairs.shape), [100, 2])


class TestInteractionDataset(unittest.TestCase):

    def setUp(self):
        self.links_file = get_data_path('links.txt')
        self.fasta_file = get_data_path('prots.fa')

        self.seqs = list(SeqIO.parse(self.fasta_file, format='fasta'))
        links = pd.read_table(self.links_file, header=None)

        truncseqs = list(map(clean, self.seqs))
        seqids = list(map(lambda x: x.id, truncseqs))
        seqdict = dict(zip(seqids, truncseqs))
        self.pairs = preprocess(seqdict, links)

    def test_sort(self):
        pass

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
        np.random.seed(1)
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
            'MILELDCGNSLIKWRVIEGAARSVAGGLAESDDALVEQLTSQQALPVRACRLVSVRSEQ'
            'ETSQLVARLEQLFPVSALVASSGKQLAGVRNGYLDYQRLGLDRWLALVAAHHLAKKACL'
            'VIDLGTAVTSDLVAADGVHLGGYICPGMTLMRSQLRTHTRRIRYDDAEARRALASLQPG'
            'QATAEAVERGCLLMLRGFVREQYAMACELLGPDCEIFLTGGDAELVRDELAGARIMPDL'
            'VFVGLALACPIE'
        )

        self.assertListEqual(list(gene), exp_gene)
        self.assertListEqual(list(pos), exp_pos)
        self.assertListEqual(list(neg), exp_neg)

    def test_iter(self):
        # Test the iter function to make sure
        # negative samples are being drawn
        np.random.seed(0)
        sampler = NegativeSampler(self.seqs)
        intsd = InteractionDataset(self.pairs, sampler)
        res = [r for r in intsd]
        self.assertEqual(len(res), self.pairs.shape[0] * intsd.num_neg)


class TestValidationDataset(unittest.TestCase):

    def setUp(self):
        self.links_file = get_data_path('links.txt')
        self.fasta_file = get_data_path('prots.fa')

        self.seqs = list(SeqIO.parse(self.fasta_file, format='fasta'))
        self.links = pd.read_table(self.links_file, header=None)

        truncseqs = list(map(clean, self.seqs))
        seqids = list(map(lambda x: x.id, truncseqs))
        seqdict = dict(zip(seqids, truncseqs))

        self.pairs = preprocess(seqdict, self.links)

    def test_getitem(self):
        np.random.seed(0)
        sampler = NegativeSampler(self.seqs)
        intsd = ValidationDataset(self.pairs, self.links, sampler)

        gene, pos, rnd, protid, taxa = intsd[0]

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
        exp_rnd = list(
            'MINEIKKEAQERMGKTLEALGHAFAKIRTGRAHPSILDSVMVSYYGADTPLRQVANVTV'
            'EDSRTLALAVFDKSMIQAVEKAIMTSDLGLNPATAGTTIRVPMPALTEETRKGYTKQAR'
            'AEAEQARVSVRNIRRDALAQLKDLQKEKEISEDEERRAGDDVQKLTDKFIGEIEKALEA'
            'KEADLMAV'
        )
        self.assertListEqual(list(gene), exp_gene)
        self.assertListEqual(list(pos), exp_pos)
        self.assertListEqual(list(rnd), exp_rnd)
        self.assertEqual(protid, '287.DR97_4286')
        self.assertEqual(taxa, 287)


    def test_iter(self):
        # Test the iter function to make sure
        # negative samples are being drawn
        np.random.seed(0)
        sampler = NegativeSampler(self.seqs)
        intsd = ValidationDataset(self.pairs, self.links, sampler)
        res = [r for r in intsd]
        self.assertEqual(len(res), self.pairs.shape[0] * intsd.num_neg)
        gene, pos, rnd, idx, taxa = list(zip(*res))
        ids = list(zip(idx, taxa))
        # make sure that if sorted, the list will be in the same order
        sorted_idx = sorted(ids, key=lambda x: (x[0], x[1]))
        self.assertListEqual(sorted_idx, ids)


class TestParse(unittest.TestCase):

    def test_parse_links(self):
        # Make sure that a validate dataloader is added
        batch_size = 1
        self.links_file = get_data_path('links.txt')
        self.fasta_file = get_data_path('prots.fa')

        res = parse(self.fasta_file, self.links_file,
                    training_column=4,
                    batch_size=batch_size,
                    num_workers=1, arm_the_gpu=False)
        self.assertEqual(len(res), 3)

        train, test, valid = res

        i = 0
        for g, p, n in train:
            i+= 1
        self.assertEqual(len(train), 83)

        i = 0
        for g, p, n in test:
            i+= 1
        self.assertEqual(len(test), 12)

        # Make sure that a validate dataloader is added
        i = 0
        for g, p, n in valid:
            i+= 1
        self.assertEqual(len(valid), 5)

    def test_parse_positive(self):
        batch_size = 1
        self.links_file = get_data_path('positive.txt')
        self.fasta_file = get_data_path('prots.fa')

        res = parse(self.fasta_file, self.links_file,
                    training_column=4,
                    batch_size=batch_size,
                    num_workers=1, arm_the_gpu=False)
        self.assertEqual(len(res), 3)
        self.assertIsNone(res[0])
        self.assertIsNone(res[1])
        self.assertIsNotNone(res[2])
        self.assertEqual(len(res[2]), 2)

    def test_parse_negative(self):
        batch_size = 1
        self.links_file = get_data_path('negative.txt')
        self.fasta_file = get_data_path('prots.fa')

        res = parse(self.fasta_file, self.links_file,
                    training_column=4,
                    batch_size=batch_size,
                    num_workers=1, arm_the_gpu=False)
        self.assertEqual(len(res), 3)
        self.assertIsNone(res[0])
        self.assertIsNotNone(res[1])
        self.assertIsNotNone(res[2])
        self.assertEqual(len(res[2]), 2)


if __name__ == "__main__":
    unittest.main()
