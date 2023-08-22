CXX=g++
IDIR=$(PWD)/include
ODIR=$(PWD)/obj
SDIR=$(PWD)/src
DNNSRC=$(SDIR)/DenseNeuralNetwork.cpp
DNNOBJ=$(ODIR)/DenseNeuralNetwork.a
CXXFLAGS=-O3 -Wall -Werror -g -std=c++17 -I$(IDIR)

.PHONY: all
all: $(DNNOBJ)

$(DNNOBJ): $(DNNSRC)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

.PHONY: clean
clean:
	rm -rf $(ODIR)/*
