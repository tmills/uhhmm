

all:  config/myconfig.ini data/simplewiki_d1_tagwords.ints.txt
	python3 scripts/d1trainer.py $<

debug: config/debug.ini
	python3 -m pdb scripts/d1trainer.py $<

short: config/debug.ini
	python3 scripts/d1trainer.py $<

profile: config/profile.ini
	ln -s beam_sampler.pyx scripts/beam_sampler.py
	/sw/bin/kernprof -l -v scripts/d1trainer.py $<
	unlink scripts/beam_sampler.py
	
config/myconfig.ini:  config/d1train.ini
	cp $< $@

data/simplewiki_d1.txt: data/simplewiki-20140903-pages-articles.wsj02to21-comparativized-gcg15-1671-4sm.fullberk.parsed.100000onward.100000first.bd.linetrees
	cat $< | ./scripts/extract_d1_trees.sh | ./scripts/trees2words.sh > $@

data/simplewiki_d1_tagwords.txt: data/simplewiki-20140903-pages-articles.wsj02to21-comparativized-gcg15-1671-4sm.fullberk.parsed.100000onward.100000first.bd.linetrees
	cat $< | grep -v "#" | ./scripts/extract_d1_trees.sh | grep "^(S" | ./scripts/trees2poswords.sh > $@

data/simplewiki_d2_tagwords.txt: data/simplewiki-20140903-pages-articles.wsj02to21-comparativized-gcg15-1671-4sm.fullberk.parsed.100000onward.100000first.bd.linetrees
	cat $< ./scripts/extract_d2_trees.sh | grep "^(S" | ./scripts/trees2poswords.sh | sort -R --random-source /dev/zero | head -5000 > $@ 

data/%.ints.txt: data/%.txt
	cat $< | perl scripts/wordFile2IntFile.pl data/$*.dict > $@

data/%.small.txt: data/%.txt
	head -100 $< > $@

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
	python3 scripts/ltf2tagwords.py $(shell cat user-lorelei-location.txt)/REFLEX_Hungarian_LDC2015E82_V1.1/data/annotation/pos_tagged/ltf > $@

data/urdu_tagwords.txt: user-lorelei-location.txt
	python3 scripts/ltf2tagwords.py $(shell cat user-lorelei-location.txt)/REFLEX_Urdu_LDC2015E14_V1.1/data/annotation/pos_tagged > $@

data/thailtf_tagwords.txt: user-lorelei-location.txt
	python3 scripts/ltf2tagwords.py $(shell cat user-lorelei-location.txt)/REFLEX_Thai_LDC2015E84_V1.1/data/annotation/pos_tagged/ltf > $@

data/tamiltf_tagwords.txt: user-lorelei-location.txt
	python3 scripts/ltf2tagwords.py $(shell cat user-lorelei-location.txt)/REFLEX_Tamil_LDC2015E83_V1.1/data/annotation/pos_tagged/ltf > $@

#convert the output to bracketed trees. '.txt' is the output file, '.origSents' is the file of the original sentences
%.brackets: MB=$(shell cat user-modelblocks-location.txt)
%.brackets: %.txt user-modelblocks-location.txt
	cat $< | python ./scripts/uhhmm2efabp.py | PYTHONPATH=$(MB)/gcg/scripts python3 $(MB)/lcparse/scripts/efabpout2linetrees.py  | sed 's/\^.,.//g;s/\^g//g;s/\_[0-9]*//g;s/\([^+ ]\)+\([^+ ]\)/\1-\2/g;' | sed 's/\([^+ ]\)+\([^+ ]\)/\1-\2/g;'  |  perl $(MB)/lcparse/scripts/remove-at-cats.pl  >  $@

############################
# Targets for building input files for morphologically-rich languages (tested on Korean wikipedia)
############################

.PRECIOUS: data/%.morf.txt genmodel/%.morf.model

data/%.morf.txt: data/%.txt genmodel/%.morf.model
	cat $< | morfessor-segment -l genmodel/$*.morf.model - | perl scripts/morf2sents.pl > $@


genmodel/%.morf.model: data/%.txt
	morfessor-train -s $@ $^



