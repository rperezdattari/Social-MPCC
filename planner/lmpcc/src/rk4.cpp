# include <cmath>
# include <cstdlib>
# include <ctime>
# include <iomanip>
# include <iostream>

using namespace std;

# include "rk4.hpp"

//****************************************************************************80

void rk4 ( void dydt ( double t, double u[], double f[] ), double tspan[2], 
  double y0[], int n, int m, double t[], double y[] )

//****************************************************************************80
//
//  Purpose:
 //
//    rk4 approximates an ODE using a Runge-Kutta fourth order method.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    22 April 2020
//
//  Author:
//
//    John Burkardt
//
//  Input:
//
//    double DYDT ( double T, double U ), a function which evaluates
//    the derivative, or right hand side of the problem.
//
//    double TSPAN[2]: the initial and final times
//
//    double Y0[M]: the initial condition
//
//    int N: the number of steps to take.
//
//    int M: the number of variables.
//
//  Output:
//
//    double t[n+1], y[(n+1)*m]: the times and solution values.
//
{
  double dt;
  double *f0;
  double *f1;
  double *f2;
  double *f3;
  int i;
  int j;
  double t0;
  double t1;
  double t2;
  double t3;
  double *u0;
  double *u1;
  double *u2;
  double *u3;

  f0 = new double[m];
  f1 = new double[m];
  f2 = new double[m];
  f3 = new double[m];
  u0 = new double[m];
  u1 = new double[m];
  u2 = new double[m];
  u3 = new double[m];

  dt = ( tspan[1] - tspan[0] ) / ( double ) ( n );

  j = 0;
  t[0] = tspan[0];
  for ( i = 0; i < m; i++ )
  {
    y[i+j*m] = y0[i];
  }

  for ( j = 0; j < n; j++ )
  {
    t0 = t[j];
    for ( i = 0; i < m; i++ )
    {
      u0[i] = y[i+j*m];
    }
    dydt ( t0, u0, f0 );

    t1 = t0 + dt / 2.0;
    for ( i = 0; i < m; i++ )
    {
      u1[i] = u0[i] + dt * f0[i] / 2.0;
    }
    dydt ( t1, u1, f1 );

    t2 = t0 + dt / 2.0;
    for ( i = 0; i < m; i++ )
    {
      u2[i] = u0[i] + dt * f1[i] / 2.0;
    }
    dydt ( t2, u2, f2 );

    t3 = t0 + dt;
    for ( i = 0; i < m; i++ )
    {
      u3[i] = u0[i] + dt * f2[i];
    }
    dydt ( t3, u3, f3 );

    t[j+1] = t[j] + dt;
    for ( i = 0; i < m; i++ )
    {
       y[i+(j+1)*m] = u0[i] + dt * ( f0[i] + 2.0 * f1[i] + 2.0 * f2[i] + f3[i] ) / 6.0;
    }
  }
/*
  Free memory.
*/
  delete [] f0;
  delete [] f1;
  delete [] f2;
  delete [] f3;
  delete [] u0;
  delete [] u1;
  delete [] u2;
  delete [] u3;

  return;
}
//****************************************************************************80

void timestamp ( )

//****************************************************************************80
//
//  Purpose:
//
//    TIMESTAMP prints the current YMDHMS date as a time stamp.
//
//  Example:
//
//    31 May 2001 09:45:54 AM
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    08 July 2009
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    None
//
{
# define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct std::tm *tm_ptr;
  std::time_t now;

  now = std::time ( NULL );
  tm_ptr = std::localtime ( &now );

  std::strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm_ptr );

  std::cout << time_buffer << "\n";

  return;
# undef TIME_SIZE
}
