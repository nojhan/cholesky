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

#include "cholesky.h"


void setformat( std::ostream& out )
{
    out << std::right;
    out << std::setfill(' ');
    out << std::setw( 5 + std::numeric_limits<double>::digits10);
    out << std::setprecision(std::numeric_limits<double>::digits10);
    out << std::setiosflags(std::ios_base::showpoint);
}


template<typename MT>
std::string format(const MT& mat )
{
    std::ostringstream out;
    setformat(out);

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


template<typename MT>
bool equal( const MT& M1, const MT& M2, double prec /* = 1/std::numeric_limits<double>::digits10 ???*/ )
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


template<typename MT >
MT error( const MT& M1, const MT& M2 )
{
    assert( M1.size1() == M2.size1() && M1.size1() == M2.size2() );

    MT Err = ublas::zero_matrix<double>(M1.size1(),M1.size2());

    for( unsigned int i=0; i<M1.size1(); ++i ) {
        for( unsigned int j=0; j<M1.size2(); ++j ) {
            Err(i,j) = M1(i,j) - M2(i,j);
        }
    }

    return Err;
}


template<typename MT >
double trigsum( const MT& M )
{
    double sum = 0;
    for( unsigned int i=0; i<M.size1(); ++i ) {
        for( unsigned int j=i; j<M.size2(); ++j ) { // triangular browsing
            sum += fabs( M(i,j) ); // absolute deviation
        }
    }
    return sum;
}


template<typename T>
double sum( const T& c )
{
     return std::accumulate(c.begin(), c.end(), 0);
}




int main(int argc, char** argv)
{
    srand(time(0));

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

    typedef double real;
    typedef cholesky::Cholesky<real>::CovarMat CovarMat;
    typedef cholesky::Cholesky<real>::FactorMat FactorMat;

    cholesky::LLT<real>     llt;
    cholesky::LLTabs<real>  llta;
    cholesky::LLTzero<real> lltz;
    cholesky::LDLT<real>    ldlt;

    unsigned int llt_fail = 0;
    std::vector<double> s0,s1,s2,s3;
    for( unsigned int n=0; n<R; ++n ) {

        // a random sample matrix
        ublas::matrix<real> S(M,N);
        for( unsigned int i=0; i<M; ++i) {
            for( unsigned int j=0; j<N; ++j) {
                S(i,j) = F * static_cast<real>(rand())/RAND_MAX;
            }
        }
        
        // a variance-covariance matrix of size N*N
        CovarMat V = ublas::prod( ublas::trans(S), S );
        assert( V.size1() == N && V.size2() == N );

        if( R == 1 ) {
            std::cout << std::endl << "Covariance matrix:" << std::endl;
            std::cout << format(V) << std::endl;
        }

        FactorMat L0; 
        try {
            L0 = llt(V);
            CovarMat V0 = ublas::prod( L0, ublas::trans(L0) );
            s0.push_back( trigsum(error(V,V0)) );
        } catch( cholesky::NotDefinitePositive & error ) {
            llt_fail++;
#ifndef NDEBUG
            std::cout << "LLT FAILED:\t" << error.what() << std::endl;
#endif
        }

        FactorMat L1 = llta(V);
        CovarMat V1 = ublas::prod( L1, ublas::trans(L1) );
        s1.push_back( trigsum(error(V,V1)) );

        FactorMat L2 = lltz(V);
        CovarMat V2 = ublas::prod( L2, ublas::trans(L2) );
        s2.push_back( trigsum(error(V,V2)) );

        FactorMat L3 = ldlt(V);
        CovarMat V3 = ublas::prod( L3, ublas::trans(L3) );
        s3.push_back( trigsum(error(V,V3)) );
    }

    std::cout << "Average error:" << std::endl;
    
    std::cout << "\tLLT:  ";
    if( s0.size() == 0 ) {
        std::cout << "NAN";
    } else {
        std::cout << sum(s0)/R;
    }
    std::cout << "\t" << llt_fail << "/" << R << " failures" << std::endl;

    std::cout << "\tLLTa: " << sum(s1)/R << std::endl;
    std::cout << "\tLLTz: " << sum(s2)/R << std::endl;
    std::cout << "\tLDLT: " << sum(s3)/R << std::endl;
}
