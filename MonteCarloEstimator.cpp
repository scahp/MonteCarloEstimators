// This code was modified from https://cameron-mcelfresh.medium.com/monte-carlo-integration-313b37157852

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <random>
#include <corecrt_math_defines.h>
#include <time.h>

template <typename T>
double MontegarloEstimate(double InLowerBound, double InUpperBound, int InIterations, T pFunc)
{
    double TotalSum = 0.0;
    double RandNumber = 0.0;
    double FunctionVal = 0.0;

    for(int i=0;i<InIterations;++i)
    {
        // Select a random number within the limits of integration
        RandNumber = InLowerBound + (float(rand()) / RAND_MAX) * (InUpperBound - InLowerBound);

        // Sample the funtion's values
        FunctionVal = pFunc(RandNumber);

        // Add the f(x) value to the running sum
        TotalSum += FunctionVal;
    }

    const double Estimate = (InUpperBound - InLowerBound) * TotalSum / InIterations;
    return Estimate;
}

template <typename T>
void MonteCarlo_Basic(T Func)
{
    double LowerBound = 1.0;
    double UpperBound = 5.0;
    int Iterations = 200;

    //auto Func = [](double x)
    //{
    //    return 10 * exp(-5 * pow(x - 3, 4));        // 12.12295737874928
    //};
    //printf("Function Approximation : %f\n", 12.12295737874928);

    double Estimate = MontegarloEstimate(LowerBound, UpperBound, Iterations, Func);

    printf("Estimate for %.1f -> %.1f is %.2f, (%d iterations)\n",
        LowerBound, UpperBound, Estimate, Iterations);
}

template <typename T>
void MonteCarlo_ReducingVariance(T Func)
{
    double LowerBound = 1.0;
    double UpperBound = 5.0;

    //auto Func = [](double x)
    //{
    //    return 10 * exp(-5 * pow(x - 3, 4));        // 12.12295737874928
    //};
    //printf("Function Approximation : %f\n", 12.12295737874928);

    for(int i=0;i<5;++i)
    {
        int Iterations = 2 * pow(4, i + 1);   // 8, 32, 128, 512, 2048

        // Monte Carlo : Estimate and StandardDeviation
        double TotalSum = 0.0;
        double TotalSumSquared = 0.0;
        for(int k=0;k<Iterations;++k)
        {
            double RandNum = LowerBound + (float(rand()) / RAND_MAX) * (UpperBound - LowerBound);
            double FunctionVal = Func(RandNum);

            TotalSum += FunctionVal;
            TotalSumSquared += pow(FunctionVal, 2);
        }

        double Estimate = (UpperBound - LowerBound) * TotalSum / Iterations;
        double Expected = TotalSum / Iterations;
        double ExpectedSquared = TotalSumSquared / Iterations;
        double StandardDeviation = (UpperBound - LowerBound)
            * sqrt((ExpectedSquared - (Expected * Expected)) / Iterations);

        printf("Estimate for %.1f -> %.1f is %.3f, StandardDeviation = %.4f, (%d iterations)\n",
            LowerBound, UpperBound, Estimate, StandardDeviation, Iterations);
    }
};

template <typename T>
void MonteCarlo_ImportanceSampling(T Func)
{
    double LowerBound = 1.0;
    double UpperBound = 5.0;
    
    //Random number generator to generate samples from the companion distribution
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(3, 1.0);

    //auto Func = [](double x) -> double
    //{
    //    return 10 * exp(-5 * pow(x - 3, 4));        // 12.12295737874928
    //};
    //printf("Function Approximation : %f\n", 12.12295737874928);

    auto PDF_Dist_3_1 = [](double x) -> double
    {
        return (1 / pow(2 * 3.14159, 0.5)) * exp(-(0.5) * pow(x - 3, 2));
    };
    
    for(int i=0;i<5;++i)
    {
        int Iterations = 2 * pow(4, i + 1);   // 8, 32, 128, 512, 2048
    
        // Monte Carlo : Estimate and StandardDeviation
        double TotalSum = 0.0;
        double TotalSumSquared = 0.0;
        for(int k=0;k<Iterations;++k)
        {
            double RandNum = distribution(generator);
            //double Weight = (1 / (UpperBound - LowerBound)) / PDF_Dist_3_1(RandNum);
            double Weight = 1.0 / PDF_Dist_3_1(RandNum);
            double FunctionVal = Func(RandNum) * Weight;
    
            TotalSum += FunctionVal;
            TotalSumSquared += pow(FunctionVal, 2);
        }
    
        // double Estimate = (UpperBound - LowerBound) * TotalSum / Iterations;
        double Estimate = TotalSum / Iterations;

        double Expected = TotalSum / Iterations;
        double ExpectedSquared = TotalSumSquared / Iterations;
        //double StandardDeviation = (UpperBound - LowerBound)
        //    * sqrt((ExpectedSquared - (Expected * Expected)) / Iterations);
        double StandardDeviation = sqrt((ExpectedSquared - (Expected * Expected)) / Iterations);

        printf("Estimate for %.1f -> %.1f is %.3f, StandardDeviation = %.4f, (%d iterations)\n",
            LowerBound, UpperBound, Estimate, StandardDeviation, Iterations);
    }
}

template <typename T>
void MonteCarlo_StratifiedSampling(T Func)
{
    double LowerBound = 0.0;
    double UpperBound = 20.0;

    //auto Func = [](double x)
    //{
    //    return exp(-1 * pow(x - 6, 4)) + exp(-1 * pow(x - 14, 4));
    //};
    //printf("Function Approximation : %f\n", 3.625609908221908);

    printf("Normal Monte Carlo Integration\n");
    for(int i=0;i<6;++i)
    {
        int Iterations = 4 * pow(4, i + 1);

        // Monte Carlo : Estimate and StandardDeviation
        double TotalSum = 0.0;
        double TotalSumSquared = 0.0;
        for (int k = 0; k < Iterations; ++k)
        {
            double RandNum = LowerBound + (float(rand()) / RAND_MAX) * (UpperBound - LowerBound);
            double FunctionVal = Func(RandNum);

            TotalSum += FunctionVal;
            TotalSumSquared += pow(FunctionVal, 2);
        }

        double Estimate = (UpperBound - LowerBound) * TotalSum / Iterations;
        double Expected = TotalSum / Iterations;
        double ExpectedSquared = TotalSumSquared / Iterations;
        double StandardDeviation = (UpperBound - LowerBound)
            * sqrt((ExpectedSquared - (Expected * Expected)) / Iterations);

        printf("Estimate for %.1f -> %.1f is %.3f, STD = %.4f, (%d iterations)\n",
            LowerBound, UpperBound, Estimate, StandardDeviation, Iterations);
    }

    printf("Stratified Sampling Monte Carlo Integration\n");
    constexpr int Subdomains = 4;
    for(int i=0;i<6;++i)
    {
        int Iterations = 4 * pow(4, i + 1);

        double TotalSum[Subdomains];
        double TotalSumSquared[Subdomains];
        
        // Divide the local iterations among the subdomains
        int SubIterations = int(float(Iterations) / Subdomains);
        for(int k=0;k<Subdomains;++k)
        {
            TotalSum[k] = 0;
            TotalSumSquared[k] = 0;
        }

        // Amount of change the range by each time
        double Increment = (UpperBound - LowerBound) / float(Subdomains);

        for(int seg = 0;seg<Subdomains;++seg)
        {
            double RandNum;
            double FunctionValue;
            double StartRange = LowerBound + seg * Increment;

            for(int n=0;n< SubIterations;++n)
            {
                RandNum = StartRange + (float(rand()) / RAND_MAX) * Increment;
                FunctionValue = Func(RandNum);

                TotalSum[seg] += FunctionValue;
                TotalSumSquared[seg] += pow(FunctionValue, 2);
            }
        }

        double EstimateArray[Subdomains];
        double ExpectedArray[Subdomains];
        double ExpectedSquaredArray[Subdomains];
        double StandardDeviationArray[Subdomains];

        for(int k=0;k<Subdomains;++k)
        {
            EstimateArray[k] = Increment * TotalSum[k] / SubIterations;    // for normal solve
            ExpectedArray[k] = TotalSum[k] / SubIterations;
            ExpectedSquaredArray[k] = TotalSumSquared[k] / SubIterations;

            StandardDeviationArray[k] = Increment 
                * sqrt((ExpectedSquaredArray[k] - ExpectedArray[k] * ExpectedArray[k]) / Iterations);
        }

        double Estimate = 0.0;
        double StandardDeviation = 0.0;

        for(int k=0;k<Subdomains;++k)
        {
            Estimate += EstimateArray[k];
            StandardDeviation += (Increment * Increment) * StandardDeviationArray[k] / SubIterations;
        }

        printf("Estimate for %.1f -> %.1f is %.3f, STD = %.4f, (%d iterations)\n",
            LowerBound, UpperBound, Estimate, StandardDeviation, Iterations);
    }
}

void ComparisonRimannSumAndMonteCarlo()
{
    //float r = 1.0f;
    //float SurfaceAreaOfSphere = (4.0f) * M_PI * (r * r);
    //float SurfaceAreaOfHemisphere = SurfaceAreaOfSphere / 2.0f;
    //printf("Surface area of hemisphere is %f (Radius : %f)\n", SurfaceAreaOfHemisphere, r);

    // Riemann sum of hemisphere's surface area
    float RimannSum = 0.0f;
    auto func = [](float InTheta, float InPhi)
    {
        return abs(2.0 * cos(InTheta)) * abs(pow(sin(InPhi), 10));
    };

    int RSSamples = 0;
    {
        float sampleDelta = 0.01f;

        for (float phi = sampleDelta; phi <= 2.0 * M_PI; phi += sampleDelta)
        {
            for (float theta = sampleDelta; theta <= 0.5 * M_PI; theta += sampleDelta)
            {
                RimannSum += func(theta, phi) * sin(theta);
                ++RSSamples;
            }
        }
        float Ranges = (2.0f * M_PI) * (0.5f * M_PI);  // == M_PI * M_PI
        RimannSum = RimannSum * (Ranges / float(RSSamples));
    }

    // MonteCarlo estimate of hemisphere's surface area
    float Estimate = 0.0f;
    int MCIterations = RSSamples;
    {
        auto funcRand = [](float InLowerBound, float InUpperBound)
        {
            return InLowerBound + (float(rand()) / RAND_MAX) * (InUpperBound - InLowerBound);
        };

        float TotalSum = 0.0;
        float RandNumber = 0.0;
        float FunctionVal = 0.0;

        float ThetaLower = 0.0f;
        float ThetaUpper = 0.5f * M_PI;
        float PhiLower = 0.0f;
        float PhiUpper = 2.0f * M_PI;
        for (int i = 0; i < MCIterations; ++i)
        {
            float Theta = funcRand(ThetaLower, ThetaUpper);
            float Phi = funcRand(PhiLower, PhiUpper);

            TotalSum += func(Theta, Phi) * sin(Theta);
        }

        float Ranges = (2.0f * M_PI) * (0.5f * M_PI);  // == M_PI * M_PI
        Estimate = Ranges / MCIterations * TotalSum;
    }

    //printf("Riemann sum VS MonteCarlo estimate of hemisphere's surface area\n");
    printf("Riemann sum : %f, NumOfSamples %d\n", RimannSum, RSSamples);
    printf("MonteCarlo estimate : %f, NumOfSamples %d\n", Estimate, MCIterations);
}


int main()
{
    srand(time(nullptr));

    enum eMC { MC_Basic = 0, MC_ReducingVariance, MC_ImportanceSampling, MC_StratifiedSampling, MC_ComparisonRimannSumAndMonteCarlo };

    eMC MCType = MC_ComparisonRimannSumAndMonteCarlo;

    double (*Func)(double x) = nullptr;

    switch(MCType)
    {
    case MC_StratifiedSampling:
        Func = [](double x) -> double
        {
            return exp(-1 * pow(x - 6, 4)) + exp(-1 * pow(x - 14, 4));
        };
        printf("Function Approximation : %f\n", 3.625609908221908);
        break;
    case MC_ComparisonRimannSumAndMonteCarlo:
        break;
    default:
        Func = [](double x) -> double
        {
            return 10 * exp(-5 * pow(x - 3, 4));        // 12.12295737874928
        };
        printf("Function Approximation : %f\n", 12.12295737874928);
        break;
    }

    switch(MCType)
    {
    case MC_Basic:
        MonteCarlo_Basic(Func);
        break;
    case MC_ReducingVariance:
        MonteCarlo_ReducingVariance(Func);
        break;
    case MC_ImportanceSampling:
        printf("ReducingVariance\n");
        MonteCarlo_ReducingVariance(Func);
        printf("ImportanceSampling\n");
        MonteCarlo_ImportanceSampling(Func);
        break;
    case MC_StratifiedSampling:
        MonteCarlo_StratifiedSampling(Func);
        break;
    case MC_ComparisonRimannSumAndMonteCarlo:
        ComparisonRimannSumAndMonteCarlo();
        break;
    default:
        printf("Invalid Monte Carlo type");
        break;
    }

    return 0;
}
