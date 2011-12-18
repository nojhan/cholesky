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
    Caner Candan <caner.candan@thalesgroup.com>
*/

#include <string>
#include <stdexcept>
#include <cmath>

#include <boost/numeric/ublas/symmetric.hpp>
using namespace boost::numeric;

namespace cholesky {

class NotDefinitePositive : public std::runtime_error
{
public:
    NotDefinitePositive( const std::string & what ) : std::runtime_error( what ) {}
};


/** Cholesky decomposition, given a matrix V, return a matrix L
 * such as V = L L^T (L^T being the transposed of L).
 *
 * Need a symmetric and positive definite matrix as an input, which 
 * should be the case of a non-ill-conditionned covariance matrix.
 * Thus, expect a (lower) triangular matrix.
 */
template< typename T >
class Cholesky
{
public:
    //! The covariance-matrix is symetric
    typedef ublas::symmetric_matrix< T, ublas::lower > CovarMat;

    //! The factorization matrix is triangular
    // FIXME check if triangular types behaviour is like having 0
    typedef ublas::matrix< T > FactorMat;

    /** Instanciate without computing anything, you are responsible of 
     * calling the algorithm and getting the result with operator()
     * */
    Cholesky( size_t s1 = 1, size_t s2 = 1 ) : 
        _L(ublas::zero_matrix<T>(s1,s2))
    {}

    /** Computation is made at instanciation and then cached in a member variable, 
     * use decomposition() to get the result.
     */
    Cholesky(const CovarMat& V) :
        _L(ublas::zero_matrix<T>(V.size1(),V.size2()))
    {
        (*this)( V );
    }

    /** Compute the factorization and cache the result */
    virtual void factorize( const CovarMat& V ) = 0;

    /** Compute the factorization and return the result */
    const FactorMat& operator()( const CovarMat& V )
    {
        this->factorize(V);
        return decomposition();
    }

    //! The decomposition of the covariance matrix
    const FactorMat & decomposition() const 
    {
        return _L;
    }

protected:

    /** Assert that the covariance matrix have the required properties and returns its dimension.
     *
     * Note: if compiled with NDEBUG, will not assert anything and just return the dimension.
     */
    unsigned assert_properties( const CovarMat& V )
    {
        unsigned int Vl = V.size1(); // number of lines

        // the result goes in _L
        _L = ublas::zero_matrix<T>(Vl,Vl);

#ifndef NDEBUG
        assert(Vl > 0);

        unsigned int Vc = V.size2(); // number of columns
        assert(Vc > 0);
        assert( Vl == Vc );

        // partial assert that V is semi-positive definite
        // assert that all diagonal elements are positives
        for( unsigned int i=0; i < Vl; ++i ) {
            assert( V(i,i) > 0 );
        }

        /* FIXME what is the more efficient way to check semi-positive definite? Candidates are:
         * perform the cholesky factorization
         * check if all eigenvalues are positives
         * check if all of the leading principal minors are positive
         */
#endif

        return Vl;
    }

    //! The decomposition is a (lower) symetric matrix, just like the covariance matrix
    FactorMat _L;
};


/** This standard algorithm makes use of square root and is thus subject
 * to round-off errors if the covariance matrix is very ill-conditioned.
 *
 * Compute L such that V = L L^T
 *
 * When called on ill-conditionned matrix,
 * will raise an exception before calling the square root on a negative number.
 */
template< typename T >
class LLT : public Cholesky<T>
{
public:
    virtual void factorize( const typename Cholesky<T>::CovarMat& V )
    {
        unsigned int N = assert_properties( V );

        unsigned int i=0, j=0, k;
        this->_L(0, 0) = sqrt( V(0, 0) );

        // end of the column
        for( j = 1; j < N; ++j ) {
            this->_L(j, 0) = V(0, j) / this->_L(0, 0);
        }

        // end of the matrix
        for( i = 1; i < N; ++i ) { // each column
            // diagonal
            T sum = 0.0;
            for( k = 0; k < i; ++k) {
                sum += this->_L(i, k) * this->_L(i, k);
            }

            this->_L(i,i) = L_i_i( V, i, sum );

            for( j = i + 1; j < N; ++j ) { // rows
                // one element
                sum = 0.0;
                for( k = 0; k < i; ++k ) {
                    sum += this->_L(j, k) * this->_L(i, k);
                }

                this->_L(j, i) = (V(j, i) - sum) / this->_L(i, i);

            } // for j in ]i,N[
        } // for i in [1,N[
    }

    /** The step of the standard LLT algorithm where round off errors may appear */
    inline virtual T L_i_i( const typename Cholesky<T>::CovarMat& V, const unsigned int& i, const T& sum ) const
    {
        // round-off errors may appear here
        if( V(i,i) - sum < 0 ) {
            std::ostringstream oss;
            oss << "V(" << i << "/" << V.size1() << ")=" << V(i,i) << " - sum=" << sum << "\t== " << V(i,i)-sum << " < 0 ";
            throw NotDefinitePositive(oss.str());
        }
        return sqrt( V(i,i) - sum );
    }

};


/** This standard algorithm makes use of square root but do not fail
 * if the covariance matrix is very ill-conditioned.
 * Here, we propagate the error by using the absolute value before
 * computing the square root.
 *
 * Be aware that this increase round-off errors, this is just a ugly
 * hack to avoid crash.
 */
template< typename T >
class LLTabs : public LLT<T>
{
public:
    inline virtual T L_i_i( const typename Cholesky<T>::CovarMat& V, const unsigned int& i, const T& sum ) const
    {
        /***** ugly hack *****/
        return sqrt( fabs( V(i,i) - sum) );
    }
};


/** This standard algorithm makes use of square root but do not fail
 * if the covariance matrix is very ill-conditioned.
 * Here, if the diagonal difference ir negative, we set it to zero.
 *
 * Be aware that this increase round-off errors, this is just a ugly
 * hack to avoid crash.
 */
template< typename T >
class LLTzero : public LLT<T>
{
public:
    inline virtual T L_i_i( const typename Cholesky<T>::CovarMat& V, const unsigned int& i, const T& sum ) const
    {
        T Lii;
        if(  V(i,i) - sum >= 0 ) {
            Lii = sqrt( V(i,i) - sum);
        } else {
            /***** ugly hack *****/
            Lii = 0;
        }
        return Lii;
    }
};


/** This alternative algorithm do not use square root in an inner loop,
 * but only for some diagonal elements of the matrix D.
 *
 * Computes L and D such as V = L D L^T. 
 * The factorized matrix is (L D^1/2), because V = (L D^1/2) (L D^1/2)^T
 */
template< typename T >
class LDLT : public Cholesky<T>
{
public:
    virtual void factorize( const typename Cholesky<T>::CovarMat& V )
    {
        // use "int" everywhere, because of the "j-1" operation
        int N = assert_properties( V );
        // example of an invertible matrix whose decomposition is undefined
        assert( V(0,0) != 0 ); 

        typename Cholesky<T>::FactorMat L = ublas::zero_matrix<T>(N,N);
        typename Cholesky<T>::FactorMat D = ublas::zero_matrix<T>(N,N);
        D(0,0) = V(0,0);

        for( int j=0; j<N; ++j ) { // each columns
            L(j, j) = 1;

            D(j,j) = V(j,j);
            for( int k=0; k<=j-1; ++k) { // sum
                D(j,j) -= L(j,k) * L(j,k) * D(k,k);
            }

            for( int i=j+1; i<N; ++i ) { // remaining rows

                L(i,j) = V(i,j);
                for( int k=0; k<=j-1; ++k) { // sum
                    L(i,j) -= L(i,k)*L(j,k) * D(k,k);
                }
                L(i,j) /= D(j,j);

            } // for i in rows
        } // for j in columns
        
        this->_L = root( L, D );
    }


    /** Compute the final symetric matrix: _L = L D^1/2
     * remember that V = ( L D^1/2) ( L D^1/2)^T
     * the factorization is thus L*D^1/2
     */
    inline typename Cholesky<T>::FactorMat root( typename Cholesky<T>::FactorMat& L, typename Cholesky<T>::FactorMat& D )
    {
        // fortunately, the square root of a diagonal matrix is the square 
        // root of all its elements
        typename Cholesky<T>::FactorMat sqrt_D = D;
        for(unsigned int i=0; i<D.size1(); ++i) {
            sqrt_D(i,i) = sqrt(D(i,i));
        }

        return ublas::prod( L, sqrt_D );
    }
};


/** Compute the LL^T Cholesky factorization of a matrix
 *
 * with the gaxpy level 2 algorithm without pivoting xPOTF2 used in LAPACK.
 *
 * Requires n^3/3 floating point operations.
 *
 * Reference:
 * Craig Lucas, LAPACK-Style Codes for level 2 and 3 Pivoted Cholesky Factorizations, February 5, 2004
 * Numerical Analysis Report, Manchester Centre for Computational Mathematics, Departements of Mathematics
 * ISSN 1360-1725
 */
template< typename T >
class Gaxpy : public Cholesky<T>
{
public:
    virtual void factorize( const typename Cholesky<T>::CovarMat& V )
    {
        unsigned int N = assert_properties( V );
        this->_L = V;
        for( unsigned int i=0; i<N; ++i) {
            // Level 2 BLAS equivalent pseudo-code:
            // L(i,i) = L(i,i) - L(i,1:i-1) L(i,1:i-1)^T
            T sum = 0.0;
//            for( unsigned int k=1; k<i-1; ++k) {
//            for( unsigned int k=1; k<i; ++k) {
            for( unsigned int k=0; k<i; ++k) {
//            for( unsigned int k=0; k<i-1; ++k) {
                sum += this->_L(i, k) * this->_L(i, k);
            }
            this->_L(i,i) = this->_L(i,i) - sum;

            // L(i,i) = sqrt( L(i,i) )
            this->_L(i,i) = L_i_i( this->_L(i,i) );

            // if 0 < i < N
            if( 0 < i && i < N ) {
                // L(i+1:n,i) = L(i+1:n,i) - L(i+1:n,1:i-1) L(i,1:i-1)^T
                for( unsigned int j = i+1; j < N; ++j ) {
                    sum = 0.0;
//                    for( unsigned int k = 1; k < i-1; ++k ) {
//                    for( unsigned int k = 1; k < i; ++k ) {
                    for( unsigned int k = 0; k < i; ++k ) {
//                    for( unsigned int k = 0; k < i-1; ++k ) {
                        sum += this->_L(j, k) * this->_L(i, k);
                    }
                    this->_L(j,i) = this->_L(j,i) - sum;
                }
            } // if 0 < i < N

            if( i < N ) {
                for( unsigned int j = i+1; j < N; ++j ) {
                    this->_L(j,i) = this->_L(j,i) / this->_L(i,i);
                }
            }
        } // for i in N
    }

    /** The step of the standard LLT algorithm where round off errors may appear */
    inline virtual T L_i_i( const T& lii ) const
    {
        // round-off errors may appear here
        if( lii <= 0 ) {
            std::ostringstream oss;
            oss << "L(i,i)==" << lii;
            throw NotDefinitePositive(oss.str());
        }
        return sqrt( lii );
    }
};


} // namespace cholesky
