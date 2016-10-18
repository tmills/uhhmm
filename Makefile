#################################
#
# Default item
# 
#################################

all:  config/myconfig.ini data/simplewiki_d1_tagwords.ints.txt $(THISDIR)/train.sh
	$(word 3, $^) $<

  
  
#################################
#
# COLING 2016 Repro
#
#################################

# The first target trains with punctuation,
# the second trains without
coling2016: projects/eve/a4-b4-g8-d2-fin/eve.nt.lower.nounary.nolabel.uhhmm \
            projects/eve/a4-b4-g8-d2-fin/eve.induc.uhhmm
            
# If running on a system with a GPU, use
# this target for a big speed increase
coling2016-GPU: projects/eve/a4-b4-g8-d2-P-fin/eve.nt.lower.nounary.nolabel.uhhmm \
            projects/eve/a4-b4-g8-d2-P-fin/eve.induc.uhhmm

            
            
#################################
#
# Variables and includes
# 
#################################

# Global variables
CXX:=g++
THISDIR := $(dir $(abspath $(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST))))
SCRIPTS  := $(THISDIR)/scripts
CUDA_PATH:=/usr/local/cuda
PYTHON_VERSION:=$(shell python3 --version | sed 's,Python \([0-9]\.[0-9]\)\.[0-9],python\1,')
NUMPY_INC=$(shell find /usr -name numpy | grep "${PYTHON_VERSION}" | grep "numpy/core/include" | sed 's,include/.*,include,')
#NUMPY_INC=env/lib/${PYTHON_VERSION}/site-packages/numpy/core/include 
#/usr/local/lib/python3.4/dist-packages/numpy/core/include
PY3_LOC=env/bin/${PYTHON_VERSION}
VPATH := genmodel data

# Include relevant items from the Modelblocks (MB) repository.
# On initialization, sets MB location to current directory,
# which is wrong. Includes to MB resources are ignored
# if MB location == . in order to allow core UHHMM
# recipes to execute without MB.
#
MSG1 := The current config file, 
MSG2 := , points to an incorrect location (
MSG3 := ). Fix it before re-running make.

define CONFIGWARN =

ATTENTION! I had to create $(CONFIG),
which is incorrectly set to the current directory.
Until you manually update $(CONFIG) to point
to your installation of Modelblocks
(https://github.com/modelblocks/modelblocks-release),
all recipes that require Modelblocks will
not work correctly.

endef

ifndef MAKECONFIG
CONFIG := user-modelblocks-location.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(CONFIGWARN))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
MB := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(MB))))
$(error $(MSG1)$(CONFIG)$(MSG2)$(MB)$(MSG3))
endif
endif

user-modelblocks-location.txt:
	echo '.' > $@

# Include relevant MB resources once the MB
# location has been entered in the config file.
ifdef MB
ifneq ($(MB),.)
include $(MB)/resource-general/Makefile
include $(MB)/resource-linetrees/Makefile
include $(MB)/resource-lcparse/Makefile
include $(MB)/resource-gcg/Makefile
endif
endif

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
  


#################################
#
# UHHMM core recipes
#
#################################

# One-liner to set up and run an UHHMM instance. Creates 
# user-specified project directory, generates input
# data files in it, generates a config file, and calls the 
# UHHMM learner.
# 
# The stem follows the following template:
#
#   <data-basename>.<postprocessing>.<config-params>.uhhmm
#
# where <data-basename> is the name of an input
# corpus.linetrees that must already exist in the genmodel
# directory of this repository, <postprocessing> is any
# hyphen-delimited postprocessing suffixes from Modelblocks
# (e.g. 'induc', which strips out all punctuation, unary
# branches, labels, etc. from the input), and config-params
# is a string of config parameters that differ from the
# the defaults in scripts/make_uhhmm_config.py. To see
# a breakdown of available parameters and instructions
# for how to specify them, run
#
#   python scripts/make_uhhmm_config.py -h
#
# This target requires Modelblocks.
#
/%.uhhmm: /%.ini /%.linetoks.ints.txt $(THISDIR)train.sh
	$(word 3, $^) $<

# Because of oddities in how Make handles wildcard expansion in prereqs,
# this rule is necessary to allow relative-path targets
%.uhhmm: $$(abspath $$@)
	$(info )

# Builds an UHHMM config file from parameters in the target stem
.PRECIOUS: /%.ini
/%.ini: $(SCRIPTS)/make_uhhmm_config.py
	mkdir -p /$(dir $*)
	python $< $(subst -, -,-$(lastword $(subst /, ,$(dir $*)))) -i /$*.linetoks.ints.txt -o /$(dir $*) -D /$*.linetoks.dict > $@

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
  
clean:
	rm -f scripts/*.{c,so}
  
  

#################################
#
# Source data creation and
# manipulation
#
#################################

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

%.ints.txt: %.txt
	cat $< | $(SCRIPTS)/lowercase.sh | perl $(SCRIPTS)/wordFile2IntFile.pl $*.dict > $@

%.small.txt: %.txt
	head -100 $< > $@

%.1kvocabfilter.txt: %.1kvocab %.txt
	python $(SCRIPTS)/filter_sentence_with_vocab.py $^ > $@

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

%.words.txt: %.tagwords.txt
	cat $^ | $(SCRIPTS)/tagwords2words.sh > $@

%-l10.words.txt: %.words.txt
	cat $^ | $(SCRIPTS)/words2len_words.sh 10 > $@

%-l20.words.txt: %.words.txt
	cat $^ | $(SCRIPTS)/words2len_words.sh 20 > $@

%-l3-10.words.txt: %-l10.words.txt
	cat $^ | perl -lane 'if($$#F >= 2){ print $$_; }' > $@

%-l3-20.words.txt: %-l20.words.txt
	cat $^ | perl -lane 'if($$#F >= 2){ print $$_; }' > $@

%.m2.words.txt: %.words.txt
	cat $^ | perl $(SCRIPTS)/removeInfrequent.pl 2 > $@

%.tagwords.txt: %.txt
	cat $^ | $(SCRIPTS)/trees2tagwords.sh > $@

%.lc.txt: %.txt
	cat $^ | $(SCRIPTS)/lowercase.sh > $@

%.len_gt3.txt: %.txt
	cat $^ | perl -lane 'if($$#F > 2){ print $$_; }' > $@

%.1kvocab: %.txt
	cat $^ | $(SCRIPTS)/get_top_k_words.sh 1000 > $@

%.rb.brackets: %.words.txt
	cat $^ | perl $(SCRIPTS)/words2rb.pl > $@

  

#################################
# 
# Targets for building input
# files for morphologically-rich
# languages (tested on Korean
# wikipedia)
#
#################################

.PRECIOUS: %.morf.txt genmodel/%.morf.model

%.morf.txt: %.txt genmodel/%.morf.model
	cat $< | morfessor-segment -l genmodel/$*.morf.model - | perl $(SCRIPTS)/morf2sents.pl > $@
 
# Requires Modelblocks
genmodel/%.morf.model: %.txt genmodel
	morfessor-train -s $@ $<

  

#################################
#
# Postprocessing, evaluation,
# and plotting
#
#################################

#convert the output to bracketed trees. '.txt' is the output file, '.origSents' is the file of the original sentences
%.brackets: %.txt user-modelblocks-location.txt
	cat $< | python $(SCRIPTS)/uhhmm2efabp.py | python $(LCPARSE-SCRIPTS)/efabpout2linetrees.py  | \
  sed 's/\^.,.//g;s/\^g//g;s/\_[0-9]*//g;s/\([^+ ]\)+\([^+ ]\)/\1-\2/g;' | sed 's/\([^+ ]\)+\([^+ ]\)/\1-\2/g;'  | \
  perl $(LCPARSE-SCRIPTS)/remove-at-cats.pl | python scripts/brackets_cleanup.py >  $@

# Generates linetrees from UHHMM sample output and renames the file
# appropriately for use with syneval
%.uhhmm.linetrees: last_sample$$(subst .,,$$(suffix %)).brackets$$(subst $$(word 1, $$(subst ., , $$*)),,$$(basename $$*)).linetrees
	cat $< > $@
  
# *.brackets.linetrees (ModelBlocks) = %.brackets (UHHMM)
# Used to generate linetrees from UHHMM output as input
# to syneval
%.brackets.linetrees: %.brackets
	cat $< > $@

# *.linetoks.txt (UHHMM) = %.linetoks (ModelBlocks)
# Used to generate linetoks as input to the UHHMM
.PRECIOUS: %.linetoks.txt
%.linetoks.txt: %.linetoks
	cat $< > $@

# Generates a CoNLL-style table with test tags in column 4 and gold tags in column 5
%.posgold.conll: %.conll $$(basename $$(basename %)).conll
	paste <(cut -f -3 $<) <(cut -f 4 $(word 2, $^)) <(cut -f 5- $<) | sed 's/\t\t\+//g' > $@
  
# Generates a space-delimited table of recall measures by iteration
# for each sample file in the user-supplied project directory
.PRECIOUS: /%.itermeasures
/%.itermeasures: $(SCRIPTS)/extract_recall_curve.py $$(foreach file, $$(wildcard $$(dir /%)last_sample*txt), /$$(basename $$(basename $$(basename $$*))).$$(notdir $$(basename $$(basename $$(basename $$*))))-$$(subst last_sample,,$$(basename $$(notdir $$(file))))-$$(subst .,,$$(suffix $$(basename $$(basename $$*))))$$(suffix $$(basename $$*)).diffname.syneval) 
	python $^ -m $(subst .,,$(suffix $*)) > $@

# Because of oddities in how Make handles wildcard expansion in prereqs,
# this rule is necessary to allow relative-path targets
%.itermeasures: $$(abspath $$@)
	$(info )

# Calculates and plots frequency of uses of a given depth level by iteration
# using the following template:
# 
#     <path>/<arbitrary-basename>.<depth-int>.dfreq.txt
# 
# The basename does not affect the creation of this target. Just pick something
# memorable.
# 
.PRECIOUS: /%.dfreq.itermeasures
/%.dfreq.itermeasures: $$(SCRIPTS)/extract_depth_counts.py $$(dir /%)last_sample*txt
	python $^ -d$(subst .,,$(suffix $*)) > $@

# Calculates and plots frequency of uses of depths >= a given level by iteration
# using the following template:
#
#     <path>/<arbitrary-basename>.<depth-int>.g.dfreq.txt
#
# The basename does not affect the creation of this target. Just pick something
# memorable.
#
.PRECIOUS: /%.g.dfreq.itermeasures
/%.g.dfreq.itermeasures: $$(SCRIPTS)/extract_depth_counts.py $$(dir /%)last_sample*txt
	python $^ -d$(subst .,,$(suffix $*)) -g > $@

# Because of oddities in how Make handles wildcard expansion in prereqs,
# this rule is necessary to allow relative-path targets
.PRECIOUS: %.dfreq.itermeasures
%.dfreq.itermeasures: $$(abspath $$@)
	$(info )  

.PRECIOUS: %.recall_curve.jpg
%.recall_curve.jpg: $(SCRIPTS)/plot_curve.r %.recall.itermeasures
	$^ -x Iteration -y 'Recall (%)' -h -o $@

.PRECIOUS: %.precision_curve.jpg
%.precision_curve.jpg: $(SCRIPTS)/plot_curve.r %.precision.itermeasures
	$^ -x Iteration -y 'Precision (%)' -h -o $@

.PRECIOUS: %.fscore_curve.jpg
%.fscore_curve.jpg: $(SCRIPTS)/plot_curve.r %.fscore.itermeasures
	$^ -x Iteration -y 'F-Score' -h -o $@

# Generates a plot from %logprobs.txt
.PRECIOUS: %.logprobs.jpg
%.logprobs.jpg: $(SCRIPTS)/plot_curve.r $$(dir %)logprobs.txt
	$^ -x Iteration -y 'Log Probability' -o $@

# Generates a plot from %.dfreq.txt  
.PRECIOUS: %.dfreq.jpg
%.dfreq.jpg: $(SCRIPTS)/plot_curve.r %.dfreq.itermeasures
	$^ -x Iteration -y 'D = $(subst .,,$(suffix $*)) Frequency' -h -o $@

# Generates a plot from %.g.dfreq.txt
.PRECIOUS: %.g.dfreq.jpg
%.g.dfreq.jpg: $(SCRIPTS)/plot_curve.r %.g.dfreq.itermeasures
	$^ -x Iteration -y 'D >= $(subst .,,$(suffix $*)) Frequency' -h -o $@

# Generates recall, logprob, and d2freq plots for a directory
# Target is never made, so this always runs.
%.plots: %.recall_curve.jpg %.precision_curve.jpg %.fscore_curve.jpg %.logprobs.jpg %.2.dfreq.jpg
	$(info )

# Generates recall, logprob, and d2freq plots for a directory
# Target is never made, so this always runs.
%.plots: %.precision_curve.jpg %.recall_curve.jpg %.fscore_curve.jpg %.logprobs.jpg %.2.dfreq.jpg
	$(info )
