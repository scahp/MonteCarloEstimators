// This code was modified from 
// - https://blog.demofox.org/2018/06/12/monte-carlo-integration-explanation-in-1d/
// - https://blog.demofox.org/2020/11/25/multiple-importance-sampling-in-1d/

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <random>
#include <corecrt_math_defines.h>
#include <time.h>

size_t GetSampleCount(int i)
{
    return (size_t)(pow(4, i + 1) * 1024);
}

double Lerp(double A, double B, double t)
{
	return A * (1.0 - t) + B * t;
}

void AddSampleToRunningAverage(double& average, double newValue, size_t sampleCount)
{
	// Incremental averaging: lerp from old value to new value by 1/(sampleCount+1)
	// https://blog.demofox.org/2016/08/23/incremental-averaging/
	double t = 1.0 / double(sampleCount + 1);
	average = Lerp(average, newValue, t);
}

struct VarianceUtil
{
	VarianceUtil(double InActualResult)
		: ActualAnswer(InActualResult)
	{
		Reset();
	}

	void Reset()
	{
		AverageOfEstimate = 0.0;
		CurrentVariance = 0.0;
		SampleCount = 0;
	}

	void AddEstimate(double InNewEstimate)
	{
		AddSampleToRunningAverage(AverageOfEstimate, InNewEstimate, SampleCount);

		// Variance is "The average of the squared differences from the mean"
		double difference = AverageOfEstimate - ActualAnswer;
		double differenceSqured = difference * difference;
		AddSampleToRunningAverage(CurrentVariance, differenceSqured, SampleCount);

		++SampleCount;
	}

	double GetVariance() const { return CurrentVariance; }
	double GetSTD() const { return sqrt(CurrentVariance); }
	double GetAverageOfEstimate() const { return AverageOfEstimate; }

	const double ActualAnswer;
	double AverageOfEstimate;
	double CurrentVariance;
	size_t SampleCount;
};

template <typename T>
double SimpleMonteCarlo(T Func, size_t InSampleCount = 10000, VarianceUtil* pVariance = nullptr)
{
    size_t numSamples = InSampleCount;
	double rangeMin = 0;
	double rangeMax = M_PI;

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(rangeMin, rangeMax);

	double width = rangeMax - rangeMin;

    double ySum = 0.0;
    for (size_t i = 1; i < numSamples; ++i)
    {
        double x = dist(mt);
        double y = Func(x);
        ySum += y;

		if (pVariance)
			pVariance->AddEstimate(width * y);
    }

    double yAverage = ySum / double(numSamples);
    double height = yAverage;

    return width * height;
}

template <typename T, typename P, typename I>
double GeneralMonteCarlo(T Func, P PDF, I InverseCDF, size_t InSampleCount = 10000, VarianceUtil* pVariance = nullptr)
{
    size_t numSamples = InSampleCount;

	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<double> dist(0.0, 1.0);      // 0.0~1.0 사이의 랜덤값을 뽑아냄.

    double estimateSum = 0.0;
    for (size_t i = 1; i <= numSamples; ++i)
    {
        double randNum = dist(mt);
        double x = InverseCDF(randNum);
        double y = Func(x);
        double pdf = PDF(x);
        double estimate = y / pdf;

        estimateSum += estimate;
        
        if (pVariance)
            pVariance->AddEstimate(estimate);
    }
    double estimateAverage = estimateSum / double(numSamples);
    
    return estimateAverage;
}

template <typename T, typename P, typename I>
double ImportanceSampledMonteCarlo(T Func, P PDF, I InverseCDF, size_t InNumSamples, VarianceUtil* pVariance = nullptr)
{
    return GeneralMonteCarlo(Func, PDF, InverseCDF, InNumSamples, pVariance);
}

template <typename T, typename P1, typename I1, typename P2, typename I2>
double MultipleImportanceSampledMonteCarlo(T Func, P1 PDF1, I1 InverseCDF1, P2 PDF2, I2 InverseCDF2
    , size_t InNumSamples, VarianceUtil* pVariance = nullptr)
{
	size_t numSamples = InNumSamples;

	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<double> dist(0.0, 1.0);      // 0.0~1.0 사이의 랜덤값을 뽑아냄.

	double estimateSum = 0.0;
    for (size_t i = 1; i <= numSamples; ++i)
    {
        double x1 = InverseCDF1(dist(mt));
        double y1 = Func(x1);
        double pdf11 = PDF1(x1);
        double pdf12 = PDF2(x1);

        double x2 = InverseCDF2(dist(mt));
        double y2 = Func(x2);
        double pdf21 = PDF1(x2);
        double pdf22 = PDF2(x2);

        double estimate = y1 / (pdf11 + pdf12) + y2 / (pdf21 + pdf22);
        estimateSum += estimate;

		if (pVariance)
			pVariance->AddEstimate(estimate);
    }

	double estimateAverage = estimateSum / double(numSamples);

	return estimateAverage;
}

template <typename T, typename P1, typename I1, typename P2, typename I2>
double MultipleImportanceSampledMonteCarlo_OneSampleMIS(T Func, P1 PDF1, I1 InverseCDF1, P2 PDF2, I2 InverseCDF2
    , size_t InNumSamples, VarianceUtil* pVariance = nullptr)
{
	size_t numSamples = InNumSamples;

	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<double> dist(0.0, 1.0);      // 0.0~1.0 사이의 랜덤값을 뽑아냄.

	double estimateSum = 0.0;
	for (size_t i = 1; i <= numSamples; ++i)
	{
		double x1 = InverseCDF1(dist(mt));
		//double y1 = Func(x1);
		double pdf11 = PDF1(x1);
		double pdf12 = PDF2(x1);
        double weight1 = pdf11 / (pdf11 + pdf12);

		double x2 = InverseCDF2(dist(mt));
		//double y2 = Func(x2);
		double pdf21 = PDF1(x2);
		double pdf22 = PDF2(x2);
        double weight2 = pdf22 / (pdf21 + pdf22);

        double totalWeight = weight1 + weight2;
        double weight1Chance = weight1 / totalWeight;

        // Function을 딱한번만 계산하기 때문에 Func 계산에 부담이 가는 경우 유리함.
        double estimate = dist(mt) < weight1Chance
            ? (Func(x1) / pdf11) * (weight1 / weight1Chance)
            : (Func(x2) / pdf22) * (weight2 / (1.0 - weight1Chance));

        estimateSum += estimate;

		if (pVariance)
			pVariance->AddEstimate(estimate);
	}

	double estimateAverage = estimateSum / double(numSamples);

	return estimateAverage;
}

void ComparisonRimannSumAndMonteCarlo()
{
    // Riemann sum of hemisphere, 2 dimension integration
    double actualResult = 1.54625;
    printf("[ActualResult : %lf]\n", actualResult);

    double RimannSum = 0.0;
    auto func = [](double InTheta, double InPhi) -> double
    {
        return abs(2.0 * cos(InTheta)) * abs(pow(sin(InPhi), 10));
    };

    int RSSamples = 0;
    {
        double sampleDelta = 0.01;

        for (double phi = sampleDelta; phi <= 2.0 * M_PI; phi += sampleDelta)
        {
            for (double theta = sampleDelta; theta <= 0.5 * M_PI; theta += sampleDelta)
            {
                RimannSum += func(theta, phi) * sin(theta);
                ++RSSamples;
            }
        }
        double Ranges = (2.0 * M_PI) * (0.5 * M_PI);  // == M_PI * M_PI
        RimannSum = RimannSum * (Ranges / double(RSSamples));
    }

    // MonteCarlo estimate of hemisphere, 2 dimension integration
    double Estimate = 0.0f;
    int MCIterations = RSSamples;
    {
		double ThetaLower = 0.0f;
		double ThetaUpper = 0.5f * M_PI;
		double PhiLower = 0.0f;
		double PhiUpper = 2.0f * M_PI;

		std::random_device rd;
		std::mt19937 mt(rd());
		std::uniform_real_distribution<double> distTheta(ThetaLower, ThetaUpper);
        std::uniform_real_distribution<double> distPhi(PhiLower, PhiUpper);

		double ySumSquared = 0.0;
		double ySum = 0.0;
		for (size_t i = 1; i < MCIterations; ++i)
		{
			double Theta = distTheta(mt);
            double Phi = distPhi(mt);
			double y = func(Theta, Phi) * sin(Theta);
			ySum += y;
			ySumSquared += y * y;
		}

		double yAverage = ySum / double(MCIterations);

        double Ranges = (2.0f * M_PI) * (0.5f * M_PI);  // == M_PI * M_PI
        Estimate = Ranges * yAverage;
    }

    printf("Riemann sum : \t\t%lf(Diff : %lf), \tNumOfSamples %d\n", RimannSum, abs(RimannSum - actualResult), RSSamples);
    printf("MonteCarlo estimate : \t%lf(Diff : %lf), \tNumOfSamples %d\n", Estimate, abs(Estimate - actualResult), MCIterations);
}

int main()
{
    srand((unsigned int)time(nullptr));

    // 1. SimpleMonteCarlo
    {
        double actualResult = 1.570796326794897;
        VarianceUtil variance(actualResult);
		printf("[Integration of 'sin(x) * sin(x)' between 0 and PI is %lf]\n", actualResult);
		
        auto Func = [](double x) -> double
		{
			return sin(x) * sin(x);         // 적분 대상 함수
		};

        printf("-----------SimpleMonteCarlo-----------\n");
        for(int i=0;i<5;++i)
        {
            size_t numSampleCount = GetSampleCount(i);

            variance.Reset();
            double simpleMonteCarlo = SimpleMonteCarlo(Func, numSampleCount, &variance);
            printf("Estimate : %lf(Diff : %lf), \tVariance : %lf, \tSTD : %lf\t[Samples : %zu]\n", simpleMonteCarlo
                , abs(simpleMonteCarlo - actualResult)
                , variance.GetVariance(), variance.GetSTD(), numSampleCount);
        }

		printf("\n");

        // 2. GeneralMonteCarlo
        {
            auto InverseCDF = [](double x) -> double
            {
                return x * M_PI;                // CDF 는 PDF의 적분
            };

            auto PDF = [](double x) -> double
            {
                return 1.0 / M_PI;              // 적분 범위 [0.0, M_PI]
            };

            printf("-----------GeneralMonteCarlo-----------\n");
            for (int i = 0; i < 5; ++i)
            {
                size_t numSampleCount = GetSampleCount(i);

                variance.Reset();
                double generalMonteCarlo = GeneralMonteCarlo(Func, PDF, InverseCDF, numSampleCount, &variance);
                printf("Estimate : %lf(Diff : %lf), \tVariance : %lf, \tSTD : %lf\t[Samples : %zu]\n", generalMonteCarlo
                    , abs(generalMonteCarlo - actualResult)
                    , variance.GetVariance(), variance.GetSTD(), numSampleCount);
            }
        }

        printf("\n");

        // 3. ImportanceSampled MonteCarlo
        {
            auto InverseCDF = [](double x) -> double
            {
                return 2.0 * asin(sqrt(x));     // CDF는 PDF의 적분, 즉, CDF는 sin(x) / 2.0 의 적분
            };

            auto PDF = [](double x) -> double
            {
                // sin(x) 를 PDF로 선택했고, 
                // 0~PI 구간에서 적분하면 총 2.0이 나오므로, PDF의 정의에 따라 0~PI 구간의 적분이 1.0이 되도록 정규화 시켜줌
                return sin(x) / 2.0;
            };

            printf("-----------ImportanceSampledMonteCarlo-----------\n");
			printf(" - PDF is 'sin(x) / 2.0', \t\tCDF is '2.0 * asin(sqrt(x))'\n");
            for (int i = 0; i < 5; ++i)
            {
                size_t numSampleCount = GetSampleCount(i);

				variance.Reset();
                double importanceSampledMonteCarlo = ImportanceSampledMonteCarlo(Func, PDF, InverseCDF, numSampleCount, &variance);
                printf("Estimate : %lf(Diff : %lf), \tVariance : %lf, \tSTD : %lf\t[Samples : %zu]\n", importanceSampledMonteCarlo
                    , abs(importanceSampledMonteCarlo - actualResult)
                    , variance.GetVariance(), variance.GetSTD(), numSampleCount);
            }
        }
    }
    printf("\n");

	// 4. ImportanceSampled MonteCarlo - PDF is match with Func
	{
		double actualResult = 2.0;
		VarianceUtil variance(actualResult);

		auto InverseCDF = [](double x) -> double
		{
			return 2.0 * asin(sqrt(x));     // CDF는 PDF의 적분, 즉, CDF는 sin(x) / 2.0 의 적분
		};

		auto PDF = [](double x) -> double
		{
			// sin(x) 를 PDF로 선택했고, 
			// 0~PI 구간에서 적분하면 총 2.0이 나오므로, PDF의 정의에 따라 0~PI 구간의 적분이 1.0이 되도록 정규화 시켜줌
			return sin(x) / 2.0;
		};

		auto FuncMatchWithPDF = [](double x) -> double
		{
			return sin(x);         // 적분 대상 함수
		};

		printf("-----------ImportanceSampled MonteCarlo - PDF is match with Func-----------\n");
		printf("[Integration of 'sin(x)' between 0 and PI is %lf]\n", actualResult);
		printf(" - PDF is 'sin(x) / 2.0', \t\tCDF is '2.0 * asin(sqrt(x))'\n");
		for (int i = 0; i < 5; ++i)
		{
            size_t numSampleCount = GetSampleCount(i);

			variance.Reset();
			double importanceSampledMonteCarlo = ImportanceSampledMonteCarlo(FuncMatchWithPDF, PDF, InverseCDF, numSampleCount, &variance);
			printf("Estimate : %lf(Diff : %lf), \tVariance : %lf, \tSTD : %lf\t[Samples : %zu]\n", importanceSampledMonteCarlo
				, abs(importanceSampledMonteCarlo - actualResult)
				, variance.GetVariance(), variance.GetSTD(), numSampleCount);
		}
	}

    printf("\n");

    {
        double actualResult = 6.283185307179586;
        VarianceUtil variance(actualResult);
		printf("[Integration of 'sin(x) * 2.0 * x' between 0 and PI is %lf]\n", actualResult);
		printf(" - PDF1 is 'sin(x) / 2.0', \t\tCDF1 is '2.0 * asin(sqrt(x))'\n");
		printf(" - PDF2 is 'x * 2.0 / (M_PI * M_PI)', \tCDF2 is 'M_PI * sqrt(x)'\n");

        // the PDF and inverse CDF of distributions we are using for integration
        auto PDF1 = [](double x) -> double
        {
            // normalizing y=sin(x) from 0 to pi to integrate to 1 from 0 to pi
            return sin(x) / 2.0;
        };

        auto InverseCDF1 = [](double x) -> double
        {
            // turning the PDF into a CDF, flipping x and y, and solving for y again
            return 2.0 * asin(sqrt(x));
        };

        auto PDF2 = [](double x) -> double
        {
            // normalizing y=2x from 0 to pi to integrate to 1 from 0 to pi
            return x * 2.0 / (M_PI * M_PI);
        };

        auto InverseCDF2 = [](double x) -> double
        {
            // turning the PDF into a CDF, flipping x and y, and solving for y again
            return M_PI * sqrt(x);
        };

        auto Func = [](double x) -> double
        {
            return sin(x) * 2.0 * x;
        };

        // 5. GeneralMonteCarlo with multiple functions
		printf("-----------GeneralMonteCarlo-----------\n");
		for (int i = 0; i < 5; ++i)
		{
            size_t numSampleCount = GetSampleCount(i);

            variance.Reset();
			double generalMonteCarlo = GeneralMonteCarlo(Func, PDF1, InverseCDF1, numSampleCount, &variance);
			printf("Estimate : %lf(Diff : %lf), \tVariance : %lf, \tSTD : %lf\t[Samples : %zu]\n", generalMonteCarlo
                , abs(generalMonteCarlo - actualResult)
                , variance.GetVariance(), variance.GetSTD(), numSampleCount);
		}

        printf("\n");

        // 6. MultipleImportanceSampledMonteCarlo with multiple functions
        printf("-----------MultipleImportanceSampledMonteCarlo-----------\n");
		for (int i = 0; i < 5; ++i)
		{
            size_t numSampleCount = GetSampleCount(i);
         
            variance.Reset();
            double multipleImportanceSampledMonteCarlo = MultipleImportanceSampledMonteCarlo(Func, PDF1, InverseCDF1, PDF2, InverseCDF2, numSampleCount, &variance);
            printf("Estimate : %lf(Diff : %lf), \tVariance : %lf, \tSTD : %lf\t[Samples : %zu]\n", multipleImportanceSampledMonteCarlo
                , abs(multipleImportanceSampledMonteCarlo - actualResult)
                , variance.GetVariance(), variance.GetSTD(), numSampleCount);
        }

        printf("\n");

        // 7. MultipleImportanceSampledMonteCarlo_OneSampleMIS with multiple functions
        printf("-----------MultipleImportanceSampledMonteCarlo_OneSampleMIS-----------\n");
		for (int i = 0; i < 5; ++i)
		{
            size_t numSampleCount = GetSampleCount(i);

            variance.Reset();
            double multipleImportanceSampledMonteCarlo_OneSampleMIS
                = MultipleImportanceSampledMonteCarlo_OneSampleMIS(Func, PDF1, InverseCDF1, PDF2, InverseCDF2, numSampleCount, &variance);
			printf("Estimate : %lf(Diff : %lf), \tVariance : %lf, \tSTD : %lf\t[Samples : %zu]\n", multipleImportanceSampledMonteCarlo_OneSampleMIS
                , abs(multipleImportanceSampledMonteCarlo_OneSampleMIS - actualResult)
				, variance.GetVariance(), variance.GetSTD(), numSampleCount);
        }
    }

    printf("\n");

    printf("-----------Comparison Between RimannSum and MonteCarlo-----------\n");
	ComparisonRimannSumAndMonteCarlo();

    return 0;

    return 0;
}
