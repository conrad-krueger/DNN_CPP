CXX=g++
IDIR=$(PWD)/include
ADIR=$(PWD)/obj
SDIR=$(PWD)/src
DNNSRC=$(SDIR)/DenseNeuralNetwork.cpp
DNNARCH=$(ADIR)/DenseNeuralNetwork.a
CXXFLAGS=-O3 -Wall -Werror -g -std=c++17 -I$(IDIR)

.PHONY: all
all: $(DNNARCH)

$(DNNARCH): $(DNNSRC)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

.PHONY: clean
clean:
	rm -rf $(ADIR)/*
