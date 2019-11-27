//调用openblas相关矩阵函数进行共轭梯度计算
//cblas_sgemm(order, transA, transB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC);
/*
实现C=alpha*A*B+beta*C
order 候选值 有ClasRowMajow 和ClasColMajow 这两个参数决定一维数组怎样存储在内存中,一般用ClasRowMajow
transA和transB ：表示矩阵A，B是否进行转置。候选参数 CblasTrans 和CblasNoTrans.
M：表示 A或C的行数。如果A转置，则表示转置后的行数
N：表示 B或C的列数。如果B转置，则表示转置后的列数。
K：表示 A的列数或B的行数（A的列数=B的行数）。如果A转置，则表示转置后的列数。
LDA：表示A的列数，与转置与否无关。
LDB：表示B的列数，与转置与否无关。
LDC：始终=N
*/
//


#include <cblas.h>
#include <stdio.h>
#include<iostream>
#include<math.h>
using namespace std;

int main() {

	/*
	A矩阵                         b矩阵
	9   17  9  -27                 1
	17  45  0  -45		       2
	9   0  126  9		      16
       -27 -45  9  135                 8
	*/
	const float A[16] = {9,17,9,-27,17,45,0,-45,9,0,126,9,-27,-45,9,135};
	float b[4] = {1,2,16,8};
	float x[4] = { 1,1,1,1 };
	float r[4];
	float temp[4];
	float p[4];
	float beta;
	//矩阵向量运算
	cblas_sgemv(CblasRowMajor, CblasNoTrans, 4, 4, -1, A, 4, x, 1, 0, temp, 1);
	//向量加法
	cblas_saxpy(4, 1, b, 1, temp, 1);
	memcpy(r, temp, sizeof(temp));
	memcpy(p, temp, sizeof(temp));

	int niterator = 0;
	float r1[4];
	for (int i = 0; i < 10; i++)//共轭梯度法的迭代次数小于方阵的维度,其实达不到10次就退出了
	{
		memcpy(r1, r, sizeof(r));
		float a,c[4];
		cblas_sgemv(CblasRowMajor, CblasNoTrans, 4, 4, 1, A, 4, p, 1, 0, c, 1);
		a = (cblas_sdot(4, r, 1, r, 1)) / (cblas_sdot(4, p, 1, c, 1));
		cblas_saxpy(4, a, p, 1, x, 1);
		cblas_sgemv(CblasRowMajor, CblasNoTrans, 4, 4, -1, A, 4, x, 1, 0, c, 1);
		float b1[4];//b1、b2为迭代的临时变量，防止改变b
		float b2[4];
		memcpy(b1, b, sizeof(b));
		memcpy(b2, b, sizeof(b));
		cblas_saxpy(4, 1, c, 1, b1, 1);
		memcpy(r, b1, sizeof(b1));

		cblas_sgemv(CblasRowMajor, CblasNoTrans, 4, 4, 1, A, 4, x, 1, -1, b2, 1);
		float wucha;
		wucha=(cblas_snrm2(4, b2, 1))/(cblas_snrm2(4,b,1));//计算二范数
		if (wucha < 10e-5)
		{
			niterator = i;
			break;
		}
		else
		{
			float a, b;
			a = cblas_snrm2(4, r, 1);
			b = cblas_snrm2(4, r1, 1);
			beta = (a*a) / (b*b);
			cblas_saxpby(4, 1,r,1,beta,p,1);
		}
	}
	cout << "迭代次数为:" << niterator << endl;
	cout << "方程组的解为:";
	for (int i = 0; i < 4; i++)
	{
		cout << x[i] << " " << endl;
	}

	return 0;
}
