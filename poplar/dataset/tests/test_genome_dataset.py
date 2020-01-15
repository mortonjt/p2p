import os
import random
import numpy as np
from genome_dataset import (GeneInterval, ExtractIntervals,
                            SampleGenes, MaskPeptides)
from util import distance
import unittest


class TestExtractIntervals(unittest.TestCase):

    def setUp(self):
        self.record_name = os.path.join('./data/ichnovirus.gb')

    def test_extract(self):
        self.maxDiff = None
        int_tfm = ExtractIntervals()
        res = int_tfm(self.record_name)['gene_intervals']

        exp = [
            GeneInterval(250, 796,
                         'MEIFPMDRLFKKNTMGNNIFHEIAIEGSLLMLRRVRDNVNEQMDTYLSD'
                         'TNDQGETCIVIVADRHRGHLAIELIEIFVGLGADINGTDNGGNTALHYT'
                         'VFNGDHALAEWLCQQPGINLNAANHDELTPLGLAIQLNIQSMKALLEAT'
                         'GAIFHDIESNDSDNDDDDDDDDDDDDDVSTRRHG', -1),
            GeneInterval(2240, 2810,
                         'MVNSCVLELFEGNTSAGNNIFHEIAMKGSLALLLEIRDKFDRPTDHALR'
                         'EWNGHGETCLHLVALMNRGQNAIRMIDILVELGADLNAKNHLGHTLLHY'
                         'ALENDDCELINWLLLHPEMNLSVRDYYDMQTDDDCFVEESEEEQEEEET'
                         'EETEEEEKTRVSFSAFSDDLMDFESDEFDDIPRWIDELVSIL', -1)
        ]
        self.assertListEqual(exp, res)


class TestSampleGenes(unittest.TestCase):

    def setUp(self):
        # self.record_name = os.path.join('./data/polyoma.gb')

        self.records = [
            GeneInterval(
                268, 469,
                ('MVLRQLSRQASVRVSKTWTGTKRRAQRIFIFILELLLEFCRGEDSVDGKNKSTTALPA'
                 'VKDSVKDS'), 1
            ),
            GeneInterval(
                504, 1560,
                ('MGAALALLGDLVASVSEAAAATGFSVAEIAAGEAAAAIEVQIAS'
                 'LATVEGITSTSEAIAAIGLTPQTYAVIAGAPGAIAGFAALIQTVTGISSLAQVGYRFF'
                 'SDWDHKVSTVGLYQQSGMALELFNPDEYYDILFPGVNTFVNNIQYLDPRHWGPSLFAT'
                 'ISQALWHVIRDDIPAITSQELQRRTERFFRDSLARFLEETTWTIVNAPVNFYNYIQDY'
                 'YSNLSPIRPSMVRQVAEREGTQVNFGHTYRIDDADSIQEVTQRMELRNKENVHSGEFI'
                 'EKTIAPGGANQRTAPQWMLPLLLGLYGTVTPALEAYEDGPNQKKRRVSRGSSQKAKGT'
                 'RASAKTTNKRRSRSSRS'), 1
            ),
            GeneInterval(
                861, 1560,
                ('MALELFNPDEYYDILFPGVNTFVNNIQYLDPRHWGPSLFATISQ'
                 'ALWHVIRDDIPAITSQELQRRTERFFRDSLARFLEETTWTIVNAPVNFYNYIQDYYSN'
                 'LSPIRPSMVRQVAEREGTQVNFGHTYRIDDADSIQEVTQRMELRNKENVHSGEFIEKT'
                 'IAPGGANQRTAPQWMLPLLLGLYGTVTPALEAYEDGPNQKKRRVSRGSSQKAKGTRAS'
                 'AKTTNKRRSRSSRS'), 1
            )
        ]


    def test_sample_within_operon(self):
        np.random.seed(0)
        window = 100
        samp_tfm = SampleGenes(num_sampled=3, within_prob = 1,
                               window_size=window)

        neighbor = GeneInterval(
            504, 1560,
            ('MGAALALLGDLVASVSEAAAATGFSVAEIAAGEAAAAIEVQIAS'
             'LATVEGITSTSEAIAAIGLTPQTYAVIAGAPGAIAGFAALIQTVTGISSLAQVGYRFF'
             'SDWDHKVSTVGLYQQSGMALELFNPDEYYDILFPGVNTFVNNIQYLDPRHWGPSLFAT'
             'ISQALWHVIRDDIPAITSQELQRRTERFFRDSLARFLEETTWTIVNAPVNFYNYIQDY'
             'YSNLSPIRPSMVRQVAEREGTQVNFGHTYRIDDADSIQEVTQRMELRNKENVHSGEFI'
             'EKTIAPGGANQRTAPQWMLPLLLGLYGTVTPALEAYEDGPNQKKRRVSRGSSQKAKGT'
             'RASAKTTNKRRSRSSRS'), 1
        )

        res = samp_tfm({'gene_intervals': self.records})
        self.assertIn(neighbor, res['genes'])
        self.assertIn(neighbor, res['next_genes'])

        # test against duplicates
        for i in range(2):
            self.assertNotEqual(res['genes'][i], res['next_genes'][i])
            self.assertLess(distance(res['genes'][i], res['next_genes'][i]),
                            window)


    def test_sample_within_operon(self):
        np.random.seed(0)
        window = 100
        samp_tfm = SampleGenes(num_sampled=3, within_prob = 0,
                               window_size=window)
        res = samp_tfm({'gene_intervals': self.records})
        is_random = False
        # test against duplicates
        for i in range(2):
            self.assertNotEqual(res['genes'][i], res['next_genes'][i])
            if distance(res['genes'][i], res['next_genes'][i]) > window:
                is_random = True
        self.assertTrue(is_random)


class TestMaskPeptides(unittest.TestCase):
    def setUp(self):
        self.records = [
            {
                'gene': GeneInterval(
                    268, 469,
                    ('MVLRQLSRQASVRVSKTWTGTKRRAQRIFIFILELLLEFCRGEDSVDGKNKSTTALPA'
                     'VKDSVKDS'), 1
                ),
                'next_gene': GeneInterval(
                    250, 796,
                    'MEIFPMDRLFKKNTMGNNIFHEIAIEGSLLMLRRVRDNVNEQMDTYLSD'
                    'TNDQGETCIVIVADRHRGHLAIELIEIFVGLGADINGTDNGGNTALHYT'
                    'VFNGDHALAEWLCQQPGINLNAANHDELTPLGLAIQLNIQSMKALLEAT'
                    'GAIFHDIESNDSDNDDDDDDDDDDDDDVSTRRHG', -1),
            },
            {
                'gene': GeneInterval(
                    504, 1560,
                    ('MGAALALLGDLVASVSEAAAATGFSVAEIAAGEAAAAIEVQIAS'
                     'LATVEGITSTSEAIAAIGLTPQTYAVIAGAPGAIAGFAALIQTVTGISSLAQVGYRFF'
                     'SDWDHKVSTVGLYQQSGMALELFNPDEYYDILFPGVNTFVNNIQYLDPRHWGPSLFAT'
                     'ISQALWHVIRDDIPAITSQELQRRTERFFRDSLARFLEETTWTIVNAPVNFYNYIQDY'
                     'YSNLSPIRPSMVRQVAEREGTQVNFGHTYRIDDADSIQEVTQRMELRNKENVHSGEFI'
                     'EKTIAPGGANQRTAPQWMLPLLLGLYGTVTPALEAYEDGPNQKKRRVSRGSSQKAKGT'
                     'RASAKTTNKRRSRSSRS'), 1),
                'next_gene': GeneInterval(
                    2240, 2810,
                    ('MVNSCVLELFEGNTSAGNNIFHEIAMKGSLALLLEIRDKFDRPTDHALR'
                     'EWNGHGETCLHLVALMNRGQNAIRMIDILVELGADLNAKNHLGHTLLHY'
                     'ALENDDCELINWLLLHPEMNLSVRDYYDMQTDDDCFVEESEEEQEEEET'
                     'EETEEEEKTRVSFSAFSDDLMDFESDEFDDIPRWIDELVSIL'), -1),
            },
            {
                'gene': GeneInterval(
                    861, 1560,
                    ('MALELFNPDEYYDILFPGVNTFVNNIQYLDPRHWGPSLFATISQ'
                     'ALWHVIRDDIPAITSQELQRRTERFFRDSLARFLEETTWTIVNAPVNFYNYIQDYYSN'
                     'LSPIRPSMVRQVAEREGTQVNFGHTYRIDDADSIQEVTQRMELRNKENVHSGEFIEKT'
                     'IAPGGANQRTAPQWMLPLLLGLYGTVTPALEAYEDGPNQKKRRVSRGSSQKAKGTRAS'
                     'AKTTNKRRSRSSRS'), 1),
                'next_gene': GeneInterval(
                    2240, 2810,
                    ('MVNSCVLELFEGNTSAGNNIFHEIAMKGSLALLLEIRDKFDRPTDHALR'
                     'EWNGHGETCLHLVALMNRGQNAIRMIDILVELGADLNAKNHLGHTLLHY'
                     'ALENDDCELINWLLLHPEMNLSVRDYYDMQTDDDCFVEESEEEQEEEET'
                     'EETEEEEKTRVSFSAFSDDLMDFESDEFDDIPRWIDELVSIL'), -1)
            }
        ]


    def test_mask(self):
        np.random.seed(0)
        mask_tfm = MaskPeptides(mask_prob=0.5, mutate_prob=0)
        res = mask_tfm(self.records)
        for r in res:
            self.assertContains('_', r['gene_seq'])

    def test_mutate(self):
        np.random.seed(0)
        mask_tfm = MaskPeptides(mask_prob=0.5, mutate_prob=0)
        res = mask_tfm(self.records)
        for r, o in zip(res, self.records):
            self.assertNotEqual(r['gene_seq'], o.sequence)


class TestGenomeDataset(unittest.TestCase):
    def setUp(self):
        pass

    def test_get_operon(self):
        pass

    def test_get_item(self):
        pass

    def test_iter(self):
        pass

    def test_random_peptide(self):
        pass

    def test_random_gene(self):
        pass

    def test_read_genbank(self):
        pass


if __name__ == '__main__':
    unittest.main()
