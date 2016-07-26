all: cyfrp.so
	@echo done

cyfrp.so:
	python setup.py build_ext --inplace
	mv *.c source

clean:
	rm -rf source/*
	rm -f cyfrp.so