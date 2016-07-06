
THISDIR := $(dir $(abspath $(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST))))
SCRIPTS  := $(THISDIR)/scripts

all:  config/myconfig.ini data/simplewiki_d1_tagwords.ints.txt $(THISDIR)/train.sh
	$(word 3, $^) $<

debug: config/debug.ini $(THISDIR)/debug.sh
	$(word 2, $^) $<

config/myconfig.ini: config/d1train.ini
	cp $< $@

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

data/darpa_y1eval_set0_tagwords.txt: user-lorelei-location.txt
	python3 $(SCRIPTS)/ltf2tagwords.py $(shell cat user-lorelei-location.txt)/LDC2016E57_LORELEI_IL3_Incident_Language_Pack_for_Year_1_Eval/set0/data/monolingual_text/ltf/

data/darpa_y1eval_setE_tagwords.txt: user-lorelei-location.txt
	python3 $(SCRIPTS)/ltf2tagwords.py $(shell cat user-lorelei-location.txt)/LDC2016E57_LORELEI_IL3_Incident_Language_Pack_for_Year_1_Eval/setE/data/monolingual_text/ltf/

data/darpa_y1eval_set0,E.tagwords.txt: data/darpa_y1eval_set0.tagwords.txt data/darpa_y1eval_setE.tagwords.txt
	cat $^ > $@

data/%.words.txt: data/%.tagwords.txt
	cat $^ | $(SCRIPTS)/tagwords2words.sh > $@

data/%-l10.words.txt: data/%.words.txt
	cat $^ | $(SCRIPTS)/words2l10words.sh > $@

data/%-l3-10.words.txt: data/%-l10.words.txt
	cat $^ | perl -lane 'if($$#F >= 2){ print $$_; }' > $@

data/%.m2.words.txt: data/%.words.txt
	cat $^ | perl $(SCRIPTS)/removeInfrequent.pl 2 > $@

#convert the output to bracketed trees. '.txt' is the output file, '.origSents' is the file of the original sentences
%.brackets: MB=$(shell cat user-modelblocks-location.txt)
%.brackets: %.txt user-modelblocks-location.txt
	cat $< | python $(SCRIPTS)/uhhmm2efabp.py | PYTHONPATH=$(MB)/gcg/scripts python3 $(MB)/resource-lcparse/scripts/efabpout2linetrees.py  | sed 's/\^.,.//g;s/\^g//g;s/\_[0-9]*//g;s/\([^+ ]\)+\([^+ ]\)/\1-\2/g;' | sed 's/\([^+ ]\)+\([^+ ]\)/\1-\2/g;'  |  perl $(MB)/resource-lcparse/scripts/remove-at-cats.pl  >  $@

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

