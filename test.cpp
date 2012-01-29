/*
This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

Copyright (C) 2010 Thales group
*/
/*
Authors:
    Johann Dréo <johann.dreo@thalesgroup.com>
*/

#include <iostream>
#include <limits>
#include <iomanip>
#include <numeric>
#include <string>
#include <map>
#include <vector>

#include "cholesky.h"


template<typename T>
void setformat( std::ostream& out )
{
    out << std::right;
    out << std::setfill(' ');
    out << std::setw( 5 + std::numeric_limits<T>::digits10);
    out << std::setprecision(std::numeric_limits<T>::digits10);
    out << std::setiosflags(std::ios_base::showpoint);
}


template<typename T, typename MT>
std::string format(const MT& mat )
{
    std::ostringstream out;
    setformat<T>(out);

    for( unsigned int i=0; i<mat.size1(); ++i) {
        for( unsigned int j=0; j<mat.size2(); ++j) {
            out << mat(i,j) << "\t";
        } // columns
        out << std::endl;
    } // rows

    return out.str();
}


template< typename T >
T round( T val, T prec = 1.0 )
{ 
    return (val > 0.0) ? 
        floor(val * prec + 0.5) / prec : 
         ceil(val * prec - 0.5) / prec ; 
}


template<typename T, typename MT>
bool equal( const MT& M1, const MT& M2, T prec )
{
    if( M1.size1() != M2.size1() || M1.size2() != M2.size2() ) {
        return false;
    }

    for( unsigned int i=0; i<M1.size1(); ++i ) {
        for( unsigned int j=0; j<M1.size2(); ++j ) {
            if( round(M1(i,j),prec) != round(M2(i,j),prec) ) {
                std::cout << "round(M(" << i << "," << j << "," << prec << ") == " 
                    << round(M1(i,j),prec) << " != " << round(M2(i,j),prec) << std::endl;
                return false;
            }
        }
    }

    return true;
}


template<typename T, typename MT>
MT error( const MT& M1, const MT& M2 )
{
    assert( M1.size1() == M2.size1() && M1.size1() == M2.size2() );

    MT Err = ublas::zero_matrix<T>(M1.size1(),M1.size2());

    for( unsigned int i=0; i<M1.size1(); ++i ) {
        for( unsigned int j=0; j<M1.size2(); ++j ) {
            Err(i,j) = M1(i,j) - M2(i,j);
        }
    }

    return Err;
}


template<typename T, typename MT>
T trigsum( const MT& M )
{
    T sum = 0;
    for( unsigned int i=0; i<M.size1(); ++i ) {
        for( unsigned int j=i; j<M.size2(); ++j ) { // triangular browsing
            sum += fabs( M(i,j) ); // absolute deviation
        }
    }
    return sum;
}


template<typename T,typename V>
T sum( const V& c )
{
     return std::accumulate(c.begin(), c.end(), static_cast<T>(0) );
}



template< typename T >
void test( unsigned int M, unsigned int N, unsigned int F, unsigned int R, unsigned int seed = time(0) )
{
    srand(seed);

    typedef typename cholesky::Cholesky<T>::CovarMat CovarMat;
    typedef typename cholesky::Cholesky<T>::FactorMat FactorMat;
    typedef typename std::map< std::string, cholesky::Cholesky<T>* > AlgoMap;
    
    AlgoMap algos;
    algos["LLT"]  = new cholesky::LLT<T>;
    algos["LLTa"] = new cholesky::LLTabs<T>;
    algos["LLTz"] = new cholesky::LLTzero<T>;
    algos["LDLT"] = new cholesky::LDLT<T>;
    algos["Gaxpy"] = new cholesky::Gaxpy<T>;

    // init data structures on the same keys than given algorithms
    std::map<std::string,unsigned int> fails;
    // triangular errors sum
    std::map<std::string, std::vector<T> > errors;
    for( typename AlgoMap::iterator ialgo = algos.begin(); ialgo != algos.end(); ++ialgo ) {
         fails[ialgo->first] = 0;
        errors[ialgo->first] = std::vector<T>();
    }

    for( unsigned int n=0; n<R; ++n ) {

        // a random sample matrix
        ublas::matrix<T> S(M,N);
        for( unsigned int i=0; i<M; ++i) {
            for( unsigned int j=0; j<N; ++j) {
                S(i,j) = F * static_cast<T>(rand())/RAND_MAX;
            }
        }
        
        // a variance-covariance matrix of size N*N
        // Note: a covariance matrix is necessarily semi-positive definite
        //       thus, any failure in the Cholesky factorization is due to round-off errors
        CovarMat V = ublas::prod( ublas::trans(S), S );
        assert( V.size1() == N && V.size2() == N );
#ifndef NDEBUG
        if( R == 1 ) {
            std::cout << std::endl << "Covariance matrix:" << std::endl;
            std::cout << format<T>(V) << std::endl;
        }
#endif
        for( typename AlgoMap::iterator ialgo = algos.begin(); ialgo != algos.end(); ++ialgo ) {
            // The LLT algorithm can fail on a sqrt(x<0) and throw an error
            // we thus count the failures
            FactorMat L; 
            try {
                L = (*ialgo->second)(V);
                CovarMat Vn = ublas::prod( L, ublas::trans(L) );
                errors[ialgo->first].push_back( trigsum<T>(error<T>(V,Vn)) );

            } catch( cholesky::NotDefinitePositive & error ) {
                fails[ialgo->first]++;
/*
#ifndef NDEBUG
                std::cout << "FAILED:\t" << error.what() << std::endl;
#endif
*/
            }

        } // for ialgo in algos
    } // for n in R

    for( typename AlgoMap::iterator ialgo = algos.begin(); ialgo != algos.end(); ++ialgo ) {
        std::string a = ialgo->first;

        std::cout << "\t" << a << ": (" << fails[a] << "/" << R << ")\t";
        if( errors[a].size() == 0 ) {
            std::cout << "NAN";
        } else {
            assert( errors[a].size() == R - fails[a] );
            std::cout << sum<T>(errors[a])/R;
        }
        std::cout << std::endl;
    } // for a in algos
}


int main(int argc, char** argv)
{
    unsigned int seed = time(0);
    unsigned int M = 10; // sample size
    unsigned int N = 12; // variables number
    unsigned int F = 10; // range factor
    unsigned int R = 1; // nb of repetitions

    if( argc >= 2 ) {
        M = std::atoi(argv[1]);
    }
    if( argc >= 3 ) {
        N = std::atoi(argv[2]);
    }
    if( argc >= 4 ) {
        F = std::atoi(argv[3]);
    }
    if( argc >= 5 ) {
        R = std::atoi(argv[4]);
    }

    std::clog << "Usage: test [sample size] [variables number] [random range] [repetitions]" << std::endl;
    std::clog << "\tsample size  = " << M << std::endl;
    std::clog << "\tmatrix size  = " << N << std::endl;
    std::clog << "\trandom range = " << F << std::endl;
    std::clog << "\trepetitions  = " << R << std::endl;
    std::clog << "Legend:" << std::endl;
    std::clog << "\tAlgo: (failures/runs)\tAverage error" << std::endl;

    std::cout << std::endl << "FLOAT" << std::endl;
    test<float>(M,N,F,R,seed);

    std::cout << std::endl << "DOUBLE" << std::endl;
    test<double>(M,N,F,R,seed);

    std::cout << std::endl << "LONG DOUBLE" << std::endl;
    test<long double>(M,N,F,R,seed);
}

