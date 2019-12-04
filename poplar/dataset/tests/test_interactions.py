import unittest
import numpy as np
from poplar.util import get_data_path
import pandas as pd
from Bio import SeqIO
from poplar.dataset.interactions import (
    InteractionDataset, parse, preprocess,
    clean, dictionary)


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
        self.assertListEqual(list(pairs.shape), [99, 2])


class TestInteractionDataset(unittest.TestCase):

    def setUp(self):
        self.links_file = get_data_path('links.txt')
        self.fasta_file = get_data_path('prots.fa')

        seqs = list(SeqIO.parse(self.fasta_file, format='fasta'))
        links = pd.read_table(self.links_file, header=None)

        truncseqs = list(map(clean, seqs))
        seqids = list(map(lambda x: x.id, truncseqs))
        seqdict = dict(zip(seqids, truncseqs))
        self.pairs = preprocess(seqdict, links)

    def test_random_peptide_draw(self):
        np.random.seed(0)
        intsd = InteractionDataset(self.pairs)
        seq = intsd.random_peptide_draw()
        self.assertEqual(len(seq), 200)

    def test_random_peptide(self):
        np.random.seed(0)
        intsd = InteractionDataset(self.pairs)
        seq = intsd.random_peptide()
        self.assertGreater(len(seq), 30)
        self.assertLess(len(seq), 1024)
        self.assertTrue(
            set(seq).issubset(set(dictionary.keys()))
        )
        self.assertNotIn('.', seq)

    def test_getitem(self):
        np.random.seed(0)
        intsd = InteractionDataset(self.pairs)
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
            'DDHJTVSEXGYYMBGHXOYRFZNIJUTQTFPPASDYRTTTOHABJZAKUXDLSXCAAEV'
            'FGIURPEJKYBBHJZDGXLOSAODVMZKULEGEPUDMEUIOPUDXPNVVQRFJDAFARS'
            'ECQDCKNQHVJAKSLXCZCDDSODURSOJBEKWLILCTQAWAGTOKTYINYCDCLNQII'
            'TICYUDMOAEDNLWNZNLYQYVOZQZWVTBIAEZGVNHYYPJSIWPLGPZBWMYYDSPD'
            'KMGDWFXLALIUKWLFPICTZXVTOUVDDHJJJXDOSMDJKNHHVBMCCYBFIEALCFV'
            'QIBRQDY'
        )

        # 'MDLFADAPLTLPDADLRYLPHWLDAPLASAWLLRLEQETPWEQPILRIHGEEHPTPRLV'
        # 'AWYGDPDAAYRYSGQVHRPLPWTALLGEIRERVEREVGQRVNGVLLNYYRDGQDSMGWH'
        # 'SDDEPELRRDPLVASLSLGGSRRFDLRRKGQTRIAHSLELTHGSLLVMRGATQHHWQHQ'
        # 'VAKTRRSCMPRLNLTFRLVYPQP'
        self.assertListEqual(list(gene), exp_gene)
        self.assertListEqual(list(pos), exp_pos)
        self.assertListEqual(list(neg), exp_neg)

    def test_getitem_truncate(self):
        np.random.seed(0)
        intsd = InteractionDataset(self.pairs)
        gene, pos, neg = intsd[98]
        self.assertEqual(len(gene), 1024)

    def test_parse(self):
        res = parse(self.fasta_file, self.links_file,
                    training_column='Training',
                    batch_size=10, num_workers=1, arm_the_gpu=False)
        train, test = res
        for g, p, n in train:
            assert True
        self.assertEqual(len(res), 2)


if __name__ == "__main__":
    unittest.main()
