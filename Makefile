CXX:=g++
THISDIR := $(dir $(abspath $(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST))))
SCRIPTS  := $(THISDIR)/scripts

CUDA_PATH:=/usr/local/cuda
PYTHON_VERSION:=$(shell python3 --version | sed 's,Python \([0-9]\.[0-9]\)\.[0-9],python\1,')
NUMPY_INC=$(shell find /usr -name numpy | grep "${PYTHON_VERSION}" | grep "numpy/core/include" | sed 's,include/.*,include,')
#NUMPY_INC=env/lib/${PYTHON_VERSION}/site-packages/numpy/core/include 
#/usr/local/lib/python3.4/dist-packages/numpy/core/include
PY3_LOC=env/bin/${PYTHON_VERSION}

all:  config/myconfig.ini data/simplewiki_d1_tagwords.ints.txt $(THISDIR)/train.sh
	$(word 3, $^) $<

osc:  config/myconfig.ini data/simplewiki_d1_tagwords.ints.txt $(THISDIR)/train_osc.sh
	$(word 3, $^)

debug: config/debug.ini $(THISDIR)/debug.sh
	$(word 2, $^) $<

scripts/CHmmSampler.so: gpusrc/ChmmSampler.o gpusrc/libhmm.a
	${CXX} -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-z,relro -Wl,-z,relro -g -fstack-protector-strong -Wformat -Werror=format-security -D_FORTIFY_SOURCE=2 -std=c++11 $^ -Lgpusrc/ -lhmm -L${CUDA_PATH}/lib64 -lcudart -o $@

gpusrc/ChmmSampler.o: gpusrc/CHmmSampler.cpp
	${CXX} -pthread -DNDEBUG -g -fwrapv -O2 -Wall -g -fstack-protector-strong -Wformat -Werror=format-security -D_FORTIFY_SOURCE=2 -fPIC -I$(NUMPY_INC) -I/usr/include/${PYTHON_VERSION} -I/usr/local/include/ -c $< -w -std=c++11  -L${CUDA_PATH}/lib64 -lcudart -L/usr/lib/x86_64-linux-gnu -l${PYTHON_VERSION} -Lgpusrc/ -lhmm -o $@

gpusrc/CHmmSampler.cpp: gpusrc/CHmmSampler.pyx
	cython --cplus gpusrc/CHmmSampler.pyx

gpusrc/libhmm.a: gpusrc/hmmsampler.o gpusrc/temp.o
	ar cru $@ $^
	ranlib $@ 

gpusrc/hmmsampler.o: gpusrc/temp.o
	${CUDA_PATH}/bin/nvcc -dlink -o $@ $^ -lcudart --shared -Xcompiler -fPIC -m64 -L${CUDA_PATH}/lib64 -Xlinker -rpath -Xlinker ${CUDA_PATH}/lib64

gpusrc/temp.o: gpusrc/HmmSampler.cu gpusrc/State.cu
	${CUDA_PATH}/bin/nvcc -rdc=true -c -o $@ $< -std=c++11 --shared -Xcompiler -fPIC -m64

config/myconfig.ini: config/d1train.ini
	cp $< $@

data/wsj_all_multiline.txt: user-ptb-location.txt
	cat $(shell cat user-ptb-location.txt)/treebank_3/parsed/mrg/wsj/*/* > $@

data/wsj_all.txt: data/wsj_all_multiline.txt
	cat $< | perl -pe 's/^\(/<s>(/' | perl -pe 's/\n//' | perl -pe 's/<s>/\n/g' | perl -pe 's/  +/ /g' | grep -v "^$$" > $@

data/wsj_all.tagwords.txt: data/wsj_all.txt
	cat $< | perl -pe 's/\(([^()]+) ([^()]+)\)/\1\/\2/g;s/\(\S*//g;s/\)//g;s/-NONE-\/\S*//g;s/  +/ /g;s/^ *//g' > $@

data/simplewiki_d1.txt: data/simplewiki-20140903-pages-articles.wsj02to21-comparativized-gcg15-1671-4sm.fullberk.parsed.100000onward.100000first.bd.linetrees
	cat $< | $(SCRIPTS)/extract_d1_trees.sh | $(SCRIPTS)/trees2words.sh > $@

data/simplewiki_d1.tagwords.txt: data/simplewiki-20140903-pages-articles.wsj02to21-comparativized-gcg15-1671-4sm.fullberk.parsed.100000onward.100000first.bd.linetrees
	cat $< | grep -v "#" | $(SCRIPTS)/extract_d1_trees.sh | grep "^(S" | $(SCRIPTS)/trees2poswords.sh > $@

data/simplewiki_d2_5k.tagwords.txt: data/simplewiki-20140903-pages-articles.wsj02to21-comparativized-gcg15-1671-4sm.fullberk.parsed.100000onward.100000first.bd.linetrees
	cat $< $(SCRIPTS)/extract_d2_trees.sh | grep "^(S" | $(SCRIPTS)/trees2poswords.sh | sort -R --random-source /dev/zero | head -5000 > $@ 

data/simplewiki_d2_all.tagwords.txt: data/simplewiki-20140903-pages-articles.wsj02to21-comparativized-gcg15-1671-4sm.fullberk.parsed.100000onward.100000first.bd.linetrees
	cat $< $(SCRIPTS)/extract_d2_trees.sh | $(SCRIPTS)/remove_wiki_junk.sh | $(SCRIPTS)/trees2poswords.sh > $@ 

data/simplewiki_all.tagwords.txt: data/simplewiki-20140903-pages-articles.wsj02to21-comparativized-gcg15-1671-4sm.fullberk.parsed.100000onward.100000first.bd.linetrees
	cat $< | $(SCRIPTS)/remove_wiki_junk.sh | $(SCRIPTS)/trees2poswords.sh > $@

data/%.ints.txt: data/%.txt
	cat $< | $(SCRIPTS)/lowercase.sh | perl $(SCRIPTS)/wordFile2IntFile.pl data/$*.dict > $@

data/%.small.txt: data/%.txt
	head -100 $< > $@

data/%.1kvocabfilter.txt: data/%.1kvocab data/%.txt
	python $(SCRIPTS)/filter_sentence_with_vocab.py $^ > $@

user-ptb-location.txt:
	echo '/home/tmill/mnt/r/resources/corpora/ptb/treebank_3/parsed/mrg/wsj_nps' > $@
	@echo ''
	@echo 'ATTENTION: I had to create "$@" for you, which may be wrong'
	@echo 'edit it to point at your penn treebank repository, and re-run make to continue!'
	@echo ''

user-lorelei-location.txt:
	echo '/home/corpora/original/various/lorelei' > $@
	@echo ''
	@echo 'ATTENTION: I had to create "$@" for you, which may be wrong'
	@echo 'edit it to point at your lorelei language pack repository, and re-run make to continue!'
	@echo ''

user-modelblocks-location.txt:
	echo '../modelblocks' > $@
	@echo ''
	@echo 'ATTENTION: I had to create "$@" for you, which may be wrong'
	@echo 'edit it to point at your local modelblocks, and re-run make to continue!'
	@echo ''


data/hungltf_tagwords.txt: user-lorelei-location.txt
	python3 $(SCRIPTS)/ltf2tagwords.py $(shell cat user-lorelei-location.txt)/REFLEX_Hungarian_LDC2015E82_V1.1/data/annotation/pos_tagged/ltf > $@

data/urdu_tagwords.txt: user-lorelei-location.txt
	python3 $(SCRIPTS)/ltf2tagwords.py $(shell cat user-lorelei-location.txt)/REFLEX_Urdu_LDC2015E14_V1.1/data/annotation/pos_tagged > $@

data/thailtf_tagwords.txt: user-lorelei-location.txt
	python3 $(SCRIPTS)/ltf2tagwords.py $(shell cat user-lorelei-location.txt)/REFLEX_Thai_LDC2015E84_V1.1/data/annotation/pos_tagged/ltf > $@

data/tamiltf_tagwords.txt: user-lorelei-location.txt
	python3 $(SCRIPTS)/ltf2tagwords.py $(shell cat user-lorelei-location.txt)/REFLEX_Tamil_LDC2015E83_V1.1/data/annotation/pos_tagged/ltf > $@

data/darpa_y1eval_set%.tagwords.txt: user-lorelei-location.txt
	python3 $(SCRIPTS)/ltf2tagwords.py $(shell cat user-lorelei-location.txt)/LDC2016E57_LORELEI_IL3_Incident_Language_Pack_for_Year_1_Eval/set$*/data/monolingual_text/ltf/ > $@

data/darpa_y1eval_set0,E.tagwords.txt: data/darpa_y1eval_set0.tagwords.txt data/darpa_y1eval_setE.tagwords.txt
	cat $^ > $@

data/darpa_y1eval_set0,1.tagwords.txt: data/darpa_y1eval_set0.tagwords.txt data/darpa_y1eval_set1.tagwords.txt
	cat $^ > $@

data/darpa_y1eval_set0,1,2.tagwords.txt: data/darpa_y1eval_set0.tagwords.txt data/darpa_y1eval_set1.tagwords.txt data/darpa_y1eval_set2.tagwords.txt
	cat $^ > $@

data/darpa_y1eval_set0,1,E.tagwords.txt: data/darpa_y1eval_set0.tagwords.txt data/darpa_y1eval_set1.tagwords.txt data/darpa_y1eval_setE.tagwords.txt
	cat $^ > $@

data/%.words.txt: data/%.tagwords.txt
	cat $^ | $(SCRIPTS)/tagwords2words.sh > $@

data/%-l10.words.txt: data/%.words.txt
	cat $^ | $(SCRIPTS)/words2len_words.sh 10 > $@

data/%-l20.words.txt: data/%.words.txt
	cat $^ | $(SCRIPTS)/words2len_words.sh 20 > $@

data/%-l3-10.words.txt: data/%-l10.words.txt
	cat $^ | perl -lane 'if($$#F >= 2){ print $$_; }' > $@

data/%-l3-20.words.txt: data/%-l20.words.txt
	cat $^ | perl -lane 'if($$#F >= 2){ print $$_; }' > $@

data/%.m2.words.txt: data/%.words.txt
	cat $^ | perl $(SCRIPTS)/removeInfrequent.pl 2 > $@

data/%.tagwords.txt: data/%.txt
	cat $^ | $(SCRIPTS)/trees2tagwords.sh > $@

data/%.lc.txt: data/%.txt
	cat $^ | $(SCRIPTS)/lowercase.sh > $@

data/%.len_gt3.txt: data/%.txt
	cat $^ | perl -lane 'if($$#F > 2){ print $$_; }' > $@

data/%.1kvocab: data/%.txt
	cat $^ | $(SCRIPTS)/get_top_k_words.sh 1000 > $@

data/%.rb.brackets: data/%.words.txt
	cat $^ | perl $(SCRIPTS)/words2rb.pl > $@

############################
# Targets for building input files for morphologically-rich languages (tested on Korean wikipedia)
############################

.PRECIOUS: data/%.morf.txt genmodel/%.morf.model

data/%.morf.txt: data/%.txt genmodel/%.morf.model
	cat $< | morfessor-segment -l genmodel/$*.morf.model - | perl $(SCRIPTS)/morf2sents.pl > $@

genmodel:
	mkdir -p genmodel

genmodel/%.morf.model: data/%.txt genmodel
	morfessor-train -s $@ $<


############################
# Targets for converting from brackets output to tokdeps output (for dep evaluation)
############################

# You'll need to start with a %.brackets containing
# parse trees, one sentence per line.

# Replace this with the path to wherever you put the scripts
LTREES-SCRIPTS = scripts/

#convert the output to bracketed trees. '.txt' is the output file, '.origSents' is the file of the original sentences
%.brackets: MB=$(shell cat user-modelblocks-location.txt)
%.brackets: %.txt user-modelblocks-location.txt
	cat $< | python $(SCRIPTS)/uhhmm2efabp.py | PYTHONPATH=$(MB)/gcg/scripts python3 $(MB)/resource-lcparse/scripts/efabpout2linetrees.py  | sed 's/\^.,.//g;s/\^g//g;s/\_[0-9]*//g;s/\([^+ ]\)+\([^+ ]\)/\1-\2/g;' | sed 's/\([^+ ]\)+\([^+ ]\)/\1-\2/g;'  |  perl $(MB)/resource-lcparse/scripts/remove-at-cats.pl | python scripts/brackets_cleanup.py >  $@

# rules
%.rules: %.brackets  $(LTREES-SCRIPTS)/trees2rules.pl
	cat $< | sed 's/:/-COLON-/g;s/=/-EQUALS-/g' |  perl $(word 2,$^)  >  $@

%.model: %.rules
	cat $< | sort | uniq -c | sort -nr | awk '{"wc -l $< | perl -lane '\''print $$F[0]'\''" | getline t; u = $$1; $$1 = u/t; print;}' | awk '{p = $$1; for (i=1;i<NF;i++) $$i=$$(i+1);$$NF="="; $$(NF + 1)=p; tmp=$$2;$$2=$$3;$$3=tmp;$$1="R";print;}' > $@

%.head.model: %.model $(LTREES-SCRIPTS)/rules2headmodel.py
	cat $< | PYTHONPATH=$$PYTHONPATH:../modelblocks/resource-gcg/scripts python3 $(word 2, $^) > $@
  
# generates unlabeled stanford dependencies from linetrees file
%.tokdeps: %.brackets $(LTREES-SCRIPTS)/trees2deps.py %.head.model
	cat $< | PYTHONPATH=$$PYTHONPATH:../modelblocks/resource-gcg/scripts python3 $(word 2, $^) $(word 3, $^) > $@
  
# generates shallow trees from unlabeled stanford dependencies
%.fromdeps.linetrees: MB=$(shell cat user-modelblocks-location.txt)
%.fromdeps.linetrees: %.tokdeps $(LTREES-SCRIPTS)/deps2trees.py
	cat $< | PYTHONPATH=$$PYTHONPATH:$(MB)/resource-gcg/scripts python $(word 2, $^) -f stanford > $@

clean:
	rm -f scripts/*.{c,so}
