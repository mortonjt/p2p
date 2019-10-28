import unittest
import numpy as np
from p2p.util import get_data_path
import pandas as pd
from Bio import SeqIO
from p2p.dataset import InteractionDataset, parse


class TestInteractionDataset(unittest.TestCase):

    def setUp(self):
        self.links_file = get_data_path('links.txt')
        self.fasta_file = get_data_path('prots.fa')

        self.seqs = list(SeqIO.parse(self.fasta_file, format='fasta'))
        self.links = pd.read_table(self.links_file)

    def test_constructor(self):
        intsd = InteractionDataset(self.seqs, self.links)
        self.assertEqual(len(intsd.links), 99)
        self.assertEqual(len(intsd.seqdict), 100)

    def test_random_peptide(self):
        np.random.seed(0)
        intsd = InteractionDataset(self.seqs, self.links)
        seq = intsd.random_peptide()
        self.assertEqual(len(seq), 234)

    def test_getitem(self):
        np.random.seed(0)
        intsd = InteractionDataset(self.seqs, self.links)
        gene, pos, neg = intsd[0]

        exp_gene = (
            'MINEIKKEAQERMGKTLEALGHAFAKIRTGRAHPSILDSVMVSYYGADTPLRQVANVTV'
            'EDSRTLALAVFDKSMIQAVEKAIMTSDLGLNPATAGTTIRVPMPALTEETRKGYTKQAR'
            'AEAEQARVSVRNIRRDALAQLKDLQKEKEISEDEERRAGDDVQKLTDKFIGEIEKALEA'
            'KEADLMAV'
        )
        exp_pos = (
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
        exp_neg = (
            'MTTSDLPAFWTVIPAAGVGSRMRADRPKQYLDLAGRTVIERTLDCFLEHPMLRGLVVCLA'
            'EDDPYWPGLDCAASRHVQRAAGGVERADSVLSGLLRLLELGAQADDWVLVHDAARPNLTR'
            'GDLDRLLEELAEDPVGGLLAVPARDTLKRSDRDGRVSETIDRSVVWLAYTPQMFRLGALH'
            'RALADALVAGVAITDEASAMEWAGYAPKLVEGRADNLKITTPEDLLRLQRSFPH'
        )

        self.assertEqual(gene, exp_gene)
        self.assertEqual(pos, exp_pos)
        self.assertEqual(neg, exp_neg)

    def test_parse(self):
        res = parse(self.fasta_file, self.links_file,
                    training_column='Training',
                    batch_size=10, num_workers=1, arm_the_gpu=False)
        self.assertEqual(len(res), 3)


if __name__ == "__main__":
    unittest.main()
