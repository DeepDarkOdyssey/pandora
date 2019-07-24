SRC_DIR=$(CURDIR)/pandora
SRC_FILES=$(wildcard *.py)
TEST_DIR=$(CURDIR)/tests
TMP_DIR=$(CURDIR)/tmp
STS_URL=http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz
LEXVEC_URL=https://www.dropbox.com/s/kguufyc2xcdi8yk/lexvec.enwiki%2Bnewscrawl.300d.W.pos.vectors.gz?dl=1

help:
	@echo "Makefile rules:"
	@echo ""
	@echo "make test"
	@echo "    run tests"
	@echo ""
	@echo "make clean"
	@echo "    remove .pyc files and __pycache__ folders"
	@echo ""
	@echo "make help    - This."

clean:
	find $(SRC_DIR) -name "*.pyc" | xargs rm
	find $(SRC_DIR) -name "__pycache__" | xargs rm -r

prepare:
ifeq ("$(wildcard $(TMP_DIR)/Stsbenchmark.tar.gz)","")
	wget $(STS_URL) -O $(TMP_DIR)/Stsbenchmark.tar.gz
endif

ifeq ("$(wildcard $(TMP_DIR)/stsbenchmakr)","")
	cd $(TMP_DIR); tar -xf Stsbenchmark.tar.gz
endif

ifeq ("$(wildcard $(TMP_DIR)/lexvec.enwiki+newscrawl.300d.W.pos.vectors.gz)", "")
	wget $(LEXVEC_URL) -O $(TMP_DIR)/lexvec.enwiki+newscrawl.300d.W.pos.vectors.gz
endif

ifeq ("$(wildcard $(TMP_DIR)/lexvec)", "")
	cd $(TMP_DIR); gunzip ./lexvec.enwiki+newscrawl.300d.W.pos.vectors.gz
endif

test:
	export PYTHONPATH=$(CURDIR); pytest