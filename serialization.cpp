#include <mpi.h>
#include <set>
#include <string>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/set.hpp>
#include <boost/iostreams/stream_buffer.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/back_inserter.hpp>

int main(int argc,char** argv) {

   int size, rank;

   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   if (size < 2) {
       if (rank == 0)
           std::cerr << "Require at least 2 tasks" << std::endl;
       MPI_Abort(MPI_COMM_WORLD, 1);
   }

   const int lentag=0;
   const int datatag=1;
   if (rank == 0) {
       int nums[] = {1,4,9,16};
       std::set<int> send_set(nums, nums+4);

       std::cout << "Rank " << rank << " sending set: ";
       for (std::set<int>::iterator i=send_set.begin(); i!=send_set.end(); i++)
           std::cout << *i << " ";
       std::cout << std::endl;

       // We're going to serialize into a std::string of bytes, and then send this
       std::string serial_str;
       boost::iostreams::back_insert_device<std::string> inserter(serial_str);
       boost::iostreams::stream<boost::iostreams::back_insert_device<std::string> > s(inserter);
       boost::archive::binary_oarchive send_ar(s);

       send_ar << send_set;
       s.flush();
       int len = serial_str.size();

       // Send length, then data
       MPI_Send( &len, 1, MPI_INT, 1, lentag, MPI_COMM_WORLD );
       MPI_Send( (void *)serial_str.data(), len, MPI_BYTE, 1, datatag, MPI_COMM_WORLD );
   } else if (rank == 1) {
       int len;
       MPI_Recv( &len, 1, MPI_INT, 0, lentag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

       char data[len+1];
       MPI_Recv( data, len, MPI_BYTE, 0, datatag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
       data[len] = '\0';

       boost::iostreams::basic_array_source<char> device(data, len);
       boost::iostreams::stream<boost::iostreams::basic_array_source<char> > s(device);
       boost::archive::binary_iarchive recv_ar(s);

       std::set<int> recv_set;
       recv_ar >> recv_set;

       std::cout << "Rank " << rank << " got set: ";
       for (std::set<int>::iterator i=recv_set.begin(); i!=recv_set.end(); i++)
           std::cout << *i << " ";
       std::cout << std::endl;
   }

   MPI_Finalize();
   return 0;
}