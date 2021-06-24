PROC = 6

all:
	mpic++ -o a_sgd -I Eigen/ -std=c++11 *.cpp

run:
	mpirun -n $(PROC) ./a_sgd

clean:
	rm a_sgd