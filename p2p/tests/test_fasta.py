import unittest
from Bio.Seq import Seq
from Bio import SeqIO
import os, sys, site, shutil
from p2p.fasta import FastaIndex


class TestIndex(unittest.TestCase):
    def setUp(self):
        entries = ['>testseq10_1',
                   'AGCTACT',
                   '>testseq10_2',
                   'AGCTAGCT',
                   '>testseq40_2',
                   'AAGCTAGCT',
                   '>testseq40_5',
                   'AAGCTAGCT\n'*100
                   ]
        self.fasta = "test.fa"
        self.fastaidx = "test.fai"
        self.revfasta = "rev.fa"
        open(self.fasta,'w').write('\n'.join(entries))
    def tearDown(self):
        if os.path.exists(self.fasta):
            os.remove(self.fasta)
        if os.path.exists(self.fastaidx):
            os.remove(self.fastaidx)
        if os.path.exists(self.revfasta):
            os.remove(self.revfasta)
    def testIndex(self):
        indexer = Indexer(self.fasta,self.fastaidx)
        indexer.index()
        indexer.load()

        seq = indexer.fetch("testseq10_1",1,4)
        self.assertEquals("AGCT",seq)
        seq = indexer.fetch("testseq40_5",1,13)
        self.assertEquals("AAGCTAGCTAAGC",seq)
        seq = indexer.fetch("testseq40_5",1,900)
        self.assertEquals("AAGCTAGCT"*100,seq)
    def testReverseFetch(self):
        indexer = Indexer(self.fasta,self.fastaidx)
        indexer.index()
        indexer.load()
        seq = indexer.reverse_fetch("testseq10_1",1,4)
        self.assertEquals("AGTA",seq)
        seq = indexer.reverse_fetch("testseq40_5",1,9)
        self.assertEquals("AGCTAGCTT",seq)
        seq = indexer.reverse_fetch("testseq40_5",1,900)
        self.assertEquals("AGCTAGCTT"*100,seq)

    def testTransform(self):
        indexer = Indexer(self.fasta,self.fastaidx)
        indexer.index()
        indexer.load()
        pos,_ = indexer.sixframe_to_nucleotide("testseq40_2", 5)
        self.assertEquals(pos,16)
        pos,_ = indexer.sixframe_to_nucleotide("testseq40_5", 5)
        self.assertEquals(pos,2686)
    def testGet(self):
        indexer = Indexer(self.fasta,self.fastaidx)
        indexer.index()
        indexer.load()
        self.assertEquals('AGCTACT',indexer["testseq10_1"])
        self.assertEquals('AGCTAGCT',indexer["testseq10_2"])
        self.assertEquals('AAGCTAGCT',indexer["testseq40_2"])
        self.assertEquals('AAGCTAGCT'*100,indexer["testseq40_5"])

class TestFasta(unittest.TestCase):
    def setUp(self):
        entries = ['>testseq1',
                   'AGCTACT',
                   '>testseq2',
                   'AGCTAGCT',
                   '>testseq2',
                   'AAGCTAGCT'
                   '>testseq3',
                   'AAGCTAGCT\n'*100
                   ]
        self.fasta = "test.fa"
        self.fastaidx = "test.fai"
        self.revfasta = "rev.fa"
        open(self.fasta,'w').write('\n'.join(entries))
    def tearDown(self):
        if os.path.exists(self.fasta):
            os.remove(self.fasta)
        if os.path.exists(self.fastaidx):
            os.remove(self.fastaidx)
        if os.path.exists(self.revfasta):
            os.remove(self.revfasta)
    def test1(self):
        reverse_complement(self.fasta,self.revfasta)
        seqs = [s for s in SeqIO.parse(self.revfasta,"fasta")]
        self.assertEquals(str(seqs[0].seq),"AGTAGCT")
        self.assertEquals(str(seqs[1].seq),"AGCTAGCT")
    def test2(self):
        remove_duplicates(self.fasta,self.revfasta)
        seqs = [s for s in SeqIO.parse(self.revfasta,"fasta")]
        self.assertEquals(len(seqs),2)
    def test3(self):
        seq = "ACACGCGACGCAGCGACGCAGCAGCAGCAGCA"
        newseq = format(seq,5)
        self.assertEquals(newseq,
                          '\n'.join([
                          "ACACG",
                          "CGACG",
                          "CAGCG",
                          "ACGCA",
                          "GCAGC",
                          "AGCAG",
                          "CA"])
                          )
