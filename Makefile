
all:  config/myconfig.ini
	python scripts/d1trainer.py $<

config/myconfig.ini:  config/d1train.ini
	cp $< $@


