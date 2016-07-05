/*
 * Code adapted from Nao Team Humboldt
 * Berlin - United - NaoTH-2011
 *
 *  Utility functions
 */


#include <cstdlib>
#include <cmath>

namespace Utils {


    inline float normal(const float &mean, const float &stddev)
    {
      static const float random_max= RAND_MAX+1.0;
      return stddev * std::sqrt(-2.0*std::log((rand()+1.0)/random_max))*std::sin(2.0*M_PI*rand()/random_max)+mean;
    }//end normal


    inline int random(int n) {
        assert(n > 0);
        return ((int)( (double(rand()) / RAND_MAX)  *n))%n;
    }

    inline int random(int min, int max)
    {
      return random( max - min + 1 ) + min;
    }
}
