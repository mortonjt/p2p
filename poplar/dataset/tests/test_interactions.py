import unittest
import numpy as np
from poplar.util import get_data_path
import pandas as pd
from Bio import SeqIO
from poplar.dataset.interactions import (
    InteractionDataset, ValidationDataset,
    parse, preprocess, clean, dictionary,
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
        self.assertListEqual(list(pairs.shape), [103, 2])


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
        links = pd.read_table(self.links_file, header=None)

        truncseqs = list(map(clean, self.seqs))
        seqids = list(map(lambda x: x.id, truncseqs))
        seqdict = dict(zip(seqids, truncseqs))
        negative = (links[2] == 'Negatome')
        pos_links = links.loc[~negative]
        neg_links = links.loc[negative]

        self.pos_pairs = preprocess(seqdict, pos_links)
        self.neg_pairs = preprocess(seqdict, neg_links)

    def test_getitem(self):
        np.random.seed(0)
        sampler = NegativeSampler(self.seqs)
        intsd = ValidationDataset(self.pos_pairs, self.neg_pairs, sampler)
        gene, pos, neg1, neg2, rnd = intsd[0]

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

        exp_neg1 = list(
            'VIGMLISDGSWARIPSRYHDAMLQAATRVRQRLANNLETLDRECSNNIQKAGVSIVHLT'
            'PVLGTCFRICGFDIKDAPNARLAPLLKAGSIDGFLSVHLFTWATGFYRYISYALDTKIC'
            'PAFYDMSSLGGEREGIRKLKSSRPGQAAPLDGAVFSCLGLSELAPDSGIYTLSVPFLIQ'
            'NEKMRTYFFMSVCSVLTCFGLYAKEKVVLKIASIAPARSIWETELKKLSAEWSEITGGL'
            'VSMKDLERVLHELREDLDRPFRAAGFRVITWTNAGWLSFYTRAPYASLGQLKKQTIALS'
            'SLDSS'
        )

        exp_neg2 = list(
            'PSRYHDAMLQAATRVRQRLANNLETLDRECSNNIQKAGVSIVHLTPVIGMLISDGSWAR'
            'IDAPNARLAPLLKAGSIDGFLSVHLFTWATGFYRYISYALDTKICPAVLGTCFRICGFD'
            'IKRKLKSSRPGQAAPLDGAVFSCLGLSELAPDSGIYTLSVPFLIQNEKFYDMSSLGGER'
            'EGICFGLYAKEKVVLKIASIAPARSIWETELKKLSAEWSEITGGLVSMKMRTYFFMSVC'
            'SVLTRPFRAAGFRVITWTNAGWLSFYTRAPYASLGQLKKQTIALSSLDSSDLERVLHEL'
            'REDLD'
        )
        exp_rnd = list(
            'MINEIKKEAQERMGKTLEALGHAFAKIRTGRAHPSILDSVMVSYYGADTPLRQVANVTV'
            'EDSRTLALAVFDKSMIQAVEKAIMTSDLGLNPATAGTTIRVPMPALTEETRKGYTKQAR'
            'AEAEQARVSVRNIRRDALAQLKDLQKEKEISEDEERRAGDDVQKLTDKFIGEIEKALEA'
            'KEADLMAV'
        )

        self.assertListEqual(list(gene), exp_gene)
        self.assertListEqual(list(pos), exp_pos)
        self.assertListEqual(list(neg1), exp_neg1)
        self.assertListEqual(list(neg2), exp_neg2)
        self.assertListEqual(list(rnd), exp_rnd)


    def test_iter(self):
        # Test the iter function to make sure
        # negative samples are being drawn
        np.random.seed(0)
        sampler = NegativeSampler(self.seqs)
        intsd = ValidationDataset(self.pos_pairs, self.neg_pairs, sampler)
        res = [r for r in intsd]
        self.assertEqual(len(res), self.pos_pairs.shape[0] * intsd.num_neg)


class TestParse(unittest.TestCase):
    def __init__(self):
        pass

    def test_parse(self):
        # Make sure that a validate dataloader is added
        batch_size = 1
        res = parse(self.fasta_file, self.links_file,
                    training_column='Training',
                    batch_size=batch_size,
                    num_workers=1, arm_the_gpu=False)
        self.assertEqual(len(res), 3)

        train, test, valid = res

        i = 0
        for g, p, n in train:
            i+= 1
        self.assertEqual(len(train), 82)

        i = 0
        for g, p, n in test:
            i+= 1
        self.assertEqual(len(test), 12)

        # Make sure that a validate dataloader is added
        # this will take in some negative samples from
        # http://www.russelllab.org/negatives/
        i = 0
        for g, p, n in valid:
            i+= 1
        self.assertEqual(len(valid), 25)



if __name__ == "__main__":
    unittest.main()
