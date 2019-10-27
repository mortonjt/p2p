from Bio.Seq import Seq
from Bio import SeqIO
import os, sys, site, shutil


class FastaIndex():
    def __init__(self,fasta, fastaidx, window=100):
        """ Creates a fasta index to allow for random access on files

        This develops an index similar to samtools faidx

        Parameters
        ----------
        fasta : path
            Input fasta
        fastaidx : path
            Output path for fasta index
        """
        self.fasta = fasta       #Input fasta
        self.fastaidx = fastaidx #Output fasta index
        self.faidx = {} #internal dict for storing byte offset values

    def index(self):
        """ Builds the actual fasta index"""
        idx = []
        # name = name of sequence
        # seqLen = length of sequence without newline characters
        # lineLen = number of characters per line
        # byteLen = length of sequence, including newline characters
        # myByteoff = byte offset of sequence
        name, seqLen, byteoff, myByteoff, lineLen, byteLen = None, 0, 0, 0, 0, 0
        index_out = open(self.fastaidx,'w')
        for seq in SeqIO.parse(self.fasta, 'fasta'):
            acc = seq.id
            seqLen = len(seq.seq)
            index_out.write('\t'.join(map(str, [acc, seqLen, myByteoff,
                                                lineLen, byteLen])))
            index_out.write('\n')
            seqLen = 0
            myByteoff = byteoff + lnlen
            seqLen = 0
            byteoff+=lnlen
        index_out.close()

    """ Load fasta index """
    def load(self):
        with open(self.fastaidx,'r') as handle:
            for line in handle:
                line=line.strip()
                cols=line.split('\t')

                chrom = cols[0]
                seqLen, byteOffset, lineLen, byteLen = map(int,cols[1:])
                self.faidx[chrom] = (seqLen, byteOffset, lineLen, byteLen)

    def __getitem__(self,defn):
        seqLen, byteOffset, lineLen, byteLen = self.faidx[defn]
        return self.fetch(defn, 1, seqLen)

    """ Retrieve a sequence based on fasta index """
    def fetch(self, defn, start, end):
        if len(self.faidx)==0:
            print("Empty table ...")
        assert isinstance(start, int)
        assert isinstance(end, int)

        self.fasta_handle = open(self.fasta,'r')
        seq = ""
        if not self.faidx.has_key(defn):
            raise ValueError('Chromosome %s not found in reference' % defn)
        seqLen,byteOffset,lineLen,byteLen=self.faidx[defn]
        start = start - 1
        pos = byteOffset + start / lineLen * byteLen + start % lineLen
        self.fasta_handle.seek(pos)
        while len(seq) < end - start:
            line = self.fasta_handle.readline()
            line = line.rstrip()
            seq=seq + line
        self.fasta_handle.close()
        return seq[:end - start]
