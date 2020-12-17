#include <Eigen/Core>
#include <iostream>
#include <LBFGSpp_B.h>

#include "ctime"
#include "random"
#include "iostream"

using namespace LBFGSpp;

typedef double Scalar;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

//返回范围内的随机浮点值 注意调取函数之前要调用srand(time(0));
Scalar random_double(Scalar l, Scalar t)
{
    return (t-l)*rand()*1.0/RAND_MAX + l;
}

//返回范围内的随机整数 注意调取函数之前要调用srand(time(0));
int random_int(int small, int big)
{
    return (rand() % (big - small)) + small;
}

class TestFunc
{
private:
    int m, n;
    // 普通二维数组做核矩阵
    Scalar **kernel;
    Scalar *obs;
    Scalar *tmp_sum;
    Scalar *fm;
public:
    TestFunc(int m_, int n_)
    {
        m = m_;
        n = n_;

        kernel = new Scalar *[m];
        for (int i = 0; i < m; i++)
        {
            kernel[i] = new Scalar [n];
        }
        obs = new Scalar [m];
        tmp_sum = new Scalar [m];
        fm = new Scalar[n];

        srand(time(0));

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                kernel[i][j] = 0.0;
            }
        }

        // 添加一些大数
        int tmp_id, tmp_size;
        double tmp_val;
        for (int i = 0; i < m; i++)
        {
            tmp_size = random_int(25, 35);
            for (int j = 0; j < tmp_size; j++)
            {
                tmp_id = random_int(0, n);
                tmp_val = random_double(-10, 10);

                kernel[i][tmp_id] = tmp_val;
            }
        }

        // 生成一组正演解 包含一些大值和一些小值
        int N2 = (int) n/2;
        for (int i = 0; i < N2; i++)
        {
            fm[i] = random_double(5, 10);
        }

        for (int i = N2; i < n; i++)
        {
            fm[i] = random_double(1, 2);
        }

        // 计算正演值
        for (int i = 0; i < m; i++)
        {
            obs[i] = 0.0;
            for (int j = 0; j < n; j++)
            {
                obs[i] += kernel[i][j]*fm[j];
            }
            // 添加噪声
            obs[i] += random_double(-1e-3, 1e-3);
        }
    }

    virtual ~TestFunc()
    {
        for (int i = 0; i < m; i++)
        {
            delete[] kernel[i];
        }
        delete[] kernel;
        delete[] obs;
        delete[] tmp_sum;
        delete[] fm;
    }

    void show_ret(const Vector& x)
    {
        for (int i = 0; i < n; i++)
        {
            std::cout << fm[i] << " " << x[i] << std::endl;
        }
        return;
    }

    Scalar operator()(const Vector& x, Vector& grad)
    {
        Scalar fx = 0.0;
        for (int i = 0; i < m; i++)
        {
            tmp_sum[i] = 0.0;
            for (int j = 0; j < n; j++)
            {
                tmp_sum[i] += kernel[i][j] * x[j];
            }
            tmp_sum[i] -= obs[i];
            fx += tmp_sum[i]*tmp_sum[i];
        }


        for (int j = 0; j < n; j++)
        {
            grad[j] = 0.0;
            for (int i = 0; i < m; i++)
            {
                grad[j] = grad[j] + kernel[i][j]*tmp_sum[i];
            }
        }

        return fx;
    }
};

int main()
{
    const int m = 80;
    const int n = 120;
    LBFGSBParam<Scalar> param;
    LBFGSBSolver<Scalar> solver(param);
    TestFunc fun(m, n);

    // Variable bounds
    Vector lb = Vector::Constant(n, 0.0);
    Vector ub = Vector::Constant(n, 0.0);
    int N2 = (int) n/2;
    for (int i = 0; i < N2; i++)
    {
        lb[i] = 0.0; // 对解的范围进行约束
        ub[i] = 10.0;
    }

    for (int i = N2; i < n; i++)
    {
        lb[i] = 0.0;
        ub[i] = 5.0;
    }

    Vector x = Vector::Constant(n, 0.0);
    // Make some initial values at the bounds
    for (int i = 0; i < N2; i++)
    {
        x[i] = 5.0;
    }

    for (int i = N2; i < n; i++)
    {
        x[i] = 2.5;
    }

    Scalar fx;
    int niter = solver.minimize(fun, x, fx, lb, ub);

    std::cout << "# " << niter << " iterations" << std::endl;
    //std::cout << "x = \n" << x.transpose() << std::endl;
    std::cout << "# " << "f(x) = " << fx << std::endl;
    fun.show_ret(x);

    return 0;
}
