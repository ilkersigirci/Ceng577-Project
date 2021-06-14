all:
	mpic++ -o a_sgd -I Eigen/ -std=c++11 *.cpp

clean: 
	rm a_sgd