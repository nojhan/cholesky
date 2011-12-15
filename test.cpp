
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
    double sum;
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


typedef double real;


int main(int argc, char** argv)
{
    srand(time(0));

    unsigned int N = 4; // size of matrix
    unsigned int R = 1000; // nb of repetitions

    if( argc >= 2 ) {
        N = std::atoi(argv[1]);
    }
    if( argc >= 3 ) {
        R = std::atoi(argv[2]);
    }

    std::cout << "Usage: t-cholesky [matrix size] [repetitions]" << std::endl;
    std::cout << "matrix size = " << N << std::endl;
    std::cout << "repetitions = " << R << std::endl;

    typedef cholesky::Cholesky::CovarMat CovarMat;
    typedef cholesky::Cholesky::FactorMat FactorMat;

    cholesky::LLT     llt();
    cholesky::LLTabs  llta();
    cholesky::LLTzero lltz();
    cholesky::LDLT    ldlt();

    std::vector<double> s0,s1,s2,s3;
    for( unsigned int n=0; n<R; ++n ) {

        // a variance-covariance matrix of size N*N
        CovarMat V(N,N);

        // random covariance matrix
        for( unsigned int i=0; i<N; ++i) {
            V(i,i) = std::pow(rand(),2); // variance should be >= 0
            for( unsigned int j=i+1; j<N; ++j) {
                V(i,j) = rand();
            }
        }

        FactorMat L0 = llt(V);
        CovarMat V0 = ublas::prod( L0, ublas::trans(L0) );
        s0.push_back( trigsum(error(V,V0)) );

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
    std::cout << "\tLLT:  " << sum(s0)/R << std::endl;
    std::cout << "\tLLTa: " << sum(s1)/R << std::endl;
    std::cout << "\tLLTz: " << sum(s2)/R << std::endl;
    std::cout << "\tLDLT: " << sum(s3)/R << std::endl;

