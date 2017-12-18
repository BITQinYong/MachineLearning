#include<opencv2\core\core.hpp>
#include<opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>
#include<math.h>
#include<map>
#include<iostream>
using namespace std;
using namespace cv;
/*
	计算两幅图的马氏距离
	参数：
		a——图片a
		b——图片b
	返回值：
		马氏距离
*/
double Mahalanobis(Mat a, Mat b)
{
	CV_Assert(a.channels() == 1 && b.channels() == 1);
	CV_Assert(a.depth() != sizeof(double) && b.depth() != sizeof(double));
	CV_Assert(a.rows == 1  && b.rows == 1);

	Mat M[2] = { a,b };
	Mat cov,mean;
	cv::calcCovarMatrix(M,2, cov, mean, CV_COVAR_NORMAL);
	double dis = cv::Mahalanobis(a, b, cov);
	return dis;
}

/*
	计算两幅图的颜色距离
	参数：
		a——图片a
		b——图片b
	返回值：
		距离
*/
double colordis(Mat a, Mat b)
{
	CV_Assert(a.channels() == 1 && b.channels() == 1);
	CV_Assert(a.depth() != sizeof(double) && b.depth() != sizeof(double));
	CV_Assert(a.rows == 1  && b.rows == 1);
	double dis = fabs(a.at<Vec<double, 1>>(0, 0)[0] - b.at<Vec<double, 1>>(0, 0)[0])*8.0 +
		         fabs(a.at<Vec<double, 1>>(0, 1)[0] - b.at<Vec<double, 1>>(0, 1)[0]) +
		         fabs(a.at<Vec<double, 1>>(0, 2)[0] - b.at<Vec<double, 1>>(0, 2)[0]);
	return dis;
}

double shapedis(Mat a, Mat b)
{
	CV_Assert(a.channels() == 1 && b.channels() == 1);
	CV_Assert(a.depth() != sizeof(double) && b.depth() != sizeof(double));
	//double row_l=1000, row_h=0;
	//double col_l=1000, col_h=0;
	//double k1, k2;
	//for(int i=0;i<a.rows;i++)
	//	for (int j = 0; j < a.cols; j++)
	//	{
	//		if (a.at<uchar>(i, j) > 100)
	//		{
	//			if (i < row_l)
	//				row_l = i;
	//			else if (i > row_h)
	//				row_h = i;
	//			if (j < col_l)
	//				col_l = j;
	//			else if (j > col_h)
	//				col_h = j;
	//		}
	//	}
	//k1 = (row_h - row_l) / (col_h - col_l);
	//row_l = 1000, row_h = 0;
	//col_l = 1000, col_h = 0;
	//for (int i = 0; i<b.rows; i++)
	//	for (int j = 0; j < b.cols; j++)
	//	{
	//		if (b.at<uchar>(i, j) > 100)
	//		{
	//			if (i < row_l)
	//				row_l = i;
	//			else if (i > row_h)
	//				row_h = i;
	//			if (j < col_l)
	//				col_l = j;
	//			else if (j > col_h)
	//				col_h = j;
	//		}
	//	}
	//k2 = (row_h - row_l) / (col_h - col_l);
	//double t;
	//if (k1 >= k2)
	//{
	//	t = k1 - k2;
	//}
	//else t = k2 - k1;
	//vector<vector<Point>> w1, w2;
	//vector<Vec4i> hierarchy1, hierarchy2;
	//findContours(a, w1, hierarchy1, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point());//提取轮廓元素
	//findContours(b, w2, hierarchy2, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point());//提取轮廓元素
	//double dis = matchShapes(w1[0], w2[0], CV_CONTOURS_MATCH_I3, 1.0);
	double dis = matchShapes(a, b, CV_CONTOURS_MATCH_I3, 1.0);
	//dis = dis*1.0 + t*20.0;
	return dis;
}