
#ifndef Statistics_H
#define	Statistics_H

#include <cmath>
#include <cstdlib>

/**
 * @brief The Statistics class is used for storing statistics about the input data, such as means and standard deviations
 *
 *
 */
class Statistics
{
    public:
        Statistics();

        void setDimensions(int i) { dimensions = i; }
        float getMin(int i) const;
        float getMax(int i) const;
        float getMean(int i) const;
        float getVariance(int i) const;
        float getStandardDeviation(int i) const;

    private:
        int dimensions;
};


#endif	/* Statistics_H */

