# main.py
from causalgenerator import CausalGenerator

nsamples=1000
ndims=8
nranges=2
generator=CausalGenerator('cubicspline')
data=generator.generator(nsamples,ndims,nranges)
