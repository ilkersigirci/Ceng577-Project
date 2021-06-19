all:
	mpic++ -o a_sgd -I Eigen/ -std=c++11 main.cpp

run:
	mpirun -n 2 ./a_sgd

clean:
	rm a_sgd