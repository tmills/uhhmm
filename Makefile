
all:  config/myconfig.ini
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
	cat $< | ./scripts/extract_d1_trees.sh | ./scripts/trees2poswords.sh > $@

data/%.ints.txt: data/%.txt
	cat $< | perl scripts/wordFile2IntFile.pl data/$*.dict > $@

data/%.small.txt: data/%.txt
	head -100 $< > $@

############################
# Targets for building input files for morphologically-rich languages (tested on Korean wikipedia)
############################

.PRECIOUS: data/%.morf.txt genmodel/%.morf.model

data/%.morf.txt: data/%.txt genmodel/%.morf.model
	cat $< | morfessor-segment -l genmodel/$*.morf.model - | perl scripts/morf2sents.pl > $@


genmodel/%.morf.model: data/%.txt
	morfessor-train -s $@ $^

	
