
THISDIR := $(dir $(abspath $(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST))))
SCRIPTS  := $(THISDIR)/scripts

NUMPY_INC=env/lib/python3.4/site-packages/numpy/core/include 
#/usr/local/lib/python3.4/dist-packages/numpy/core/include
PY3_LOC=env/bin/python3.4

all:  config/myconfig.ini data/simplewiki_d1_tagwords.ints.txt $(THISDIR)/train.sh
	$(word 3, $^) $<

osc:  config/myconfig.ini data/simplewiki_d1_tagwords.ints.txt $(THISDIR)/train_osc.sh
	$(word 3, $^)

debug: config/debug.ini $(THISDIR)/debug.sh
	$(word 2, $^) $<

scripts/CHmmSampler.so: gpusrc/ChmmSampler.o gpusrc/libhmm.a
	g++-4.9 -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-z,relro -Wl,-z,relro -g -fstack-protector-strong -Wformat -Werror=format-security -D_FORTIFY_SOURCE=2 -std=c++11 $^ -Lgpusrc/ -lhmm -L/usr/local/cuda/lib64 -lcudart -o $@

gpusrc/ChmmSampler.o: gpusrc/CHmmSampler.cpp
	g++-4.9 -pthread -DNDEBUG -g -fwrapv -O2 -Wall -g -fstack-protector-strong -Wformat -Werror=format-security -D_FORTIFY_SOURCE=2 -fPIC -I$(NUMPY_INC) -I/usr/include/python3.4m -I/usr/local/include/ -c $< -w -std=c++11  -L/usr/local/cuda/lib64 -lcudart -L/usr/lib/x86_64-linux-gnu -lpython3.4m -Lgpusrc/ -lhmm -o $@

gpusrc/CHmmSampler.cpp: gpusrc/CHmmSampler.pyx
	cython --cplus gpusrc/CHmmSampler.pyx

gpusrc/libhmm.a: gpusrc/hmmsampler.o gpusrc/temp.o
	ar cru $@ $^
	ranlib $@ 

gpusrc/hmmsampler.o: gpusrc/temp.o
	nvcc -dlink -o $@ $^ -lcudart --shared -Xcompiler -fPIC -m64 -L/usr/local/cuda/lib64 -Xlinker -rpath -Xlinker /usr/local/cuda/lib64

gpusrc/temp.o: gpusrc/HmmSampler.cu gpusrc/State.cu
	nvcc -rdc=true -c -o $@ $< -std=c++11 --shared -Xcompiler -fPIC -m64

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

data/simplewiki_d1_tagwords.txt: data/simplewiki-20140903-pages-articles.wsj02to21-comparativized-gcg15-1671-4sm.fullberk.parsed.100000onward.100000first.bd.linetrees
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

data/%.lc.txt: data/%.txt
	cat $^ | $(SCRIPTS)/lowercase.sh > $@

data/%.len_gt3.txt: data/%.txt
	cat $^ | perl -lane 'if($$#F > 2){ print $$_; }' > $@

data/%.1kvocab: data/%.txt
	cat $^ | $(SCRIPTS)/get_top_k_words.sh 1000 > $@

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

data/%.m2.words.txt: data/%.words.txt
	cat $^ | perl $(SCRIPTS)/removeInfrequent.pl 2 > $@

data/%.tagwords.txt: data/%.txt
	cat $^ | $(SCRIPTS)/trees2tagwords.sh > $@

#convert the output to bracketed trees. '.txt' is the output file, '.origSents' is the file of the original sentences
%.brackets: MB=$(shell cat user-modelblocks-location.txt)
%.brackets: %.txt user-modelblocks-location.txt
	cat $< | python $(SCRIPTS)/uhhmm2efabp.py | PYTHONPATH=$(MB)/gcg/scripts python3 $(MB)/lcparse/scripts/efabpout2linetrees.py  | sed 's/\^.,.//g;s/\^g//g;s/\_[0-9]*//g;s/\([^+ ]\)+\([^+ ]\)/\1-\2/g;' | sed 's/\([^+ ]\)+\([^+ ]\)/\1-\2/g;'  |  perl $(MB)/lcparse/scripts/remove-at-cats.pl | python scripts/brackets_cleanup.py >  $@

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

clean:
	rm scripts/*.{c,so}

