#include<opencv2\core\core.hpp>
#include<opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>
#include<math.h>
#include<map>
#include<iostream>
using namespace std;
using namespace cv;
/*
	���ûҶȹ������󷽷���ȡͼƬ��������
	������
		src����Դͼ��
		dis�������������
		angle����������Ƕ�(���ǻ���)
*/
Mat GLCM(Mat src, double dis, double angle)
{
	/*�����Ծ���*/
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
	����Ҷȹ��������Haralick�ĸ�����
	������
		src����ԴͼƬ
*/
vector<double> GLCM_Haralick(Mat src)
{
	CV_Assert(src.channels() == 1);
	CV_Assert(src.rows > 0 && src.cols > 0);

	/*������ض���Ҫ��ƽ��ֵ�ͷ���*/
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
	/*����Աȶȡ���ضȡ������������*/
	vector<double> Hara(4,0);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<Vec<double, 1>>(i,j)[0]>0) {
				Hara[0] += (i - j)*(i - j)*src.at<Vec<double, 1>>(i, j)[0];//�Աȶ�
				Hara[1] += (i - ux)*(j - uy)*src.at<Vec<double, 1>>(i, j)[0] / (stdx*stdy);//��ض�
				Hara[2] += src.at<Vec<double, 1>>(i, j)[0]*src.at<Vec<double, 1>>(i, j)[0];//����
				Hara[3] += src.at<Vec<double, 1>>(i, j)[0] / (1 + abs(i - j));//�����
			}
		}
	}
	return Hara;
}

