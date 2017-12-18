#include<opencv2\core\core.hpp>
#include<opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>
#include<math.h>
#include<map>
#include<iostream>
using namespace std;
using namespace cv;
/*
	利用灰度共生矩阵方法提取图片纹理特征
	参数：
		src——源图像
		dis——极坐标距离
		angle——极坐标角度(不是弧度)
*/
Mat GLCM(Mat src, double dis, double angle)
{
	/*共生对距离*/
	int dx = dis*cos(angle*CV_PI/180);
	int dy = dis*sin(angle*CV_PI/180);

	CV_Assert(dx != 0 || dy != 0);
	CV_Assert(src.channels() == 1);
	CV_Assert(src.rows > dx && src.cols > dy);

	Mat det0 = Mat::zeros(256,256,CV_16UC1);

	int xStart = dx > 0 ? 0 : abs(dx);
	int yStart = dy > 0 ? 0 : abs(dy);
	int xEnd = dx < 0 ? src.rows : src.rows - dx;
	int yEnd = dy < 0 ? src.cols : src.cols - dy;

	for (int i = xStart; i < xEnd; i++) {
		for (int j = yStart; j < yEnd; j++) {
			int x = i + dx;
			int y = j + dy;
			if (0 <= x&&x < src.rows && 0 <= y&&y < src.cols) {
				unsigned int a = src.at<Vec<uchar, 1>>(i, j)[0];
				unsigned int b = src.at<Vec<uchar, 1>>(x, y)[0];
				det0.at<Vec<unsigned short, 1>>(a, b)[0]++;
			}
		}
	}
	int num = (xEnd - xStart)*(yEnd - yStart);

	Mat det;
	det0.convertTo(det, CV_64FC1, 1.0 / num);
	
	return det;
}

/*
	计算灰度共生矩阵的Haralick四个特征
	参数：
		src——源图片
*/
vector<double> GLCM_Haralick(Mat src)
{
	CV_Assert(src.channels() == 1);
	CV_Assert(src.rows > 0 && src.cols > 0);

	/*计算相关度需要的平均值和方差*/
	Mat sumRow = Mat::zeros(src.rows, 1, CV_64FC1);
	Mat sumCol = Mat::zeros(1, src.cols, CV_64FC1);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			sumRow.at<Vec<double, 1>>(i, 0) += src.at<Vec<double, 1>>(i, j);
			sumCol.at<Vec<double, 1>>(0, j) += src.at<Vec<double, 1>>(i, j);
		}
	}
	double ux = 0, uy = 0, stdx = 0, stdy = 0;
	for (int i = 0; i < src.rows; i++) {
		ux += i*sumRow.at<Vec<double, 1>>(i, 0)[0];
	}
	for (int i = 0; i < src.rows; i++) {
		stdx += (i - ux)*(i - ux)*sumRow.at<Vec<double, 1>>(i, 0)[0];
	}
	for (int j = 0; j < src.cols; j++) {
		uy += j*sumCol.at<Vec<double, 1>>(0, j)[0];
	}
	for (int j = 0; j < src.cols; j++) {
		stdy += (j - uy)*(j - uy)*sumCol.at<Vec<double, 1>>(0, j)[0];
	}
	/*计算对比度、相关度、能量、逆矩阵*/
	vector<double> Hara(4,0);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<Vec<double, 1>>(i,j)[0]>0) {
				Hara[0] += (i - j)*(i - j)*src.at<Vec<double, 1>>(i, j)[0];//对比度
				Hara[1] += (i - ux)*(j - uy)*src.at<Vec<double, 1>>(i, j)[0] / (stdx*stdy);//相关度
				Hara[2] += src.at<Vec<double, 1>>(i, j)[0]*src.at<Vec<double, 1>>(i, j)[0];//能量
				Hara[3] += src.at<Vec<double, 1>>(i, j)[0] / (1 + abs(i - j));//逆矩阵
			}
		}
	}
	return Hara;
}

