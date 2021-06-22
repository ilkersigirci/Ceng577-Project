all:
	mpic++ -o a_sgd -I Eigen/ -std=c++11 *.cpp

run:
	mpirun -n 6 ./a_sgd

clean:
	rm a_sgd