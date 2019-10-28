import unittest
import numpy as np
from p2p.util import get_data_path
import pandas as pd
from Bio import SeqIO
from p2p.dataset import InteractionDataset


class TestInteractionDataset(unittest.TestCase):

    def setUp(self):
        self.links_file = get_data_path('links.txt')
        self.fasta_file = get_data_path('prots.fa')

    def test_constructor(self):
        intsd = InteractionDataset(self.fasta_file, self.links_file)
        self.assertEqual(len(intsd.links), 99)
        self.assertEqual(len(intsd.seqdict), 100)

    def test_random_peptide(self):
        np.random.seed(0)
        intsd = InteractionDataset(self.fasta_file, self.links_file)
        seq = intsd.random_peptide()
        self.assertEqual(len(seq), 234)

    def test_getitem(self):
        np.random.seed(0)
        intsd = InteractionDataset(self.fasta_file, self.links_file)
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

    # def test_iter(self):
    #     intsd = InteractionDataset(self.fasta_file, self.links_file)
    #     seq = next(intsd)
    #     self.assertEqual(len(seq), 203)
    #     exp = ('MSVELWQQCVDLLRDELPSQQFNTWIRPLQVEAEGDELRVYAPNRFVLDWVNE'
    #            'KYLGRLLELLGERGEGQLPALSLLIGSKRSRTPRAAIVPSQTHVAPPPPVAPP'
    #            'PAPVQPVSAAPVVVPREELPPVTTAPSVSSDPYEPEEPSIDPLAAAMPAGAAP'
    #            'AVRTERNVQVEGALKHTSYLNRTFTFENFVEGKSNQLARAAAWQVADNLKHGY'
    #            'NPLFLYGGVGLGKTHLMHAVGNHLLKKNPNAKVVYLHSERFVADMVKALQLNA'
    #            'INEFKRFYRSVDALLIDDIQFFARKERSQEEFFHTFNALLEGGQQVILTSDRY'
    #            'PKEIEGLEERLKSRFGWGLTVAVEPPELETRVAILMKKAEQAKIELPHDAAFF'
    #            'IAQRIRSNVRELEGALKRVIAHSHFMGRPITIELIRESLKDLLALQDKLVSID'
    #            'NIQRTVAEYYKIKISDLLSKRRSRSVARPRQVAMALSKELTNHSLPEIGVAFG'
    #            'GRDHTTVLHACRKIAQLRESDADIREDYKNLLRTLTT')
    #     self.assertEqual(str(seq.seq), exp)


if __name__ == "__main__":
    unittest.main()
