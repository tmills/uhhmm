
all:  config/myconfig.ini
	python3 scripts/d1trainer.py $<

config/myconfig.ini:  config/d1train.ini
	cp $< $@

data/good_sents.txt: data/simplewiki-20140903-pages-articles.wsj02to21-comparativized-gcg15-1671-4sm.fullberk.parsed.100000onward.100000first.bd.linetrees
	cat $< | ./scripts/extract_d1_trees.sh | ./scripts/trees2words.sh > $@

data/good_sents_ints.txt: data/good_sents.txt
	cat $< | perl scripts/wordFile2IntFile.pl /dev/null > $@

data/good_sents_tags.txt: data/simplewiki-20140903-pages-articles.wsj02to21-comparativized-gcg15-1671-4sm.fullberk.parsed.100000onward.100000first.bd.linetrees
	cat $< | ./scripts/extract_d1_trees.sh | ./scripts/trees2pos.sh > $@

data/good_sents_tags_ints.txt: data/good_sents_tags.txt
	cat $< | perl scripts/wordFile2IntFile.pl /dev/null > $@

data/good_sents_tagwords.txt: data/simplewiki-20140903-pages-articles.wsj02to21-comparativized-gcg15-1671-4sm.fullberk.parsed.100000onward.100000first.bd.linetrees
	cat $< | ./scripts/extract_d1_trees.sh | ./scripts/trees2poswords.sh > $@

data/good_sents_tagwords_ints.txt:data/good_sents_tagwords.txt
	cat $< | perl scripts/wordFile2IntFile.pl data/dict.txt > $@
