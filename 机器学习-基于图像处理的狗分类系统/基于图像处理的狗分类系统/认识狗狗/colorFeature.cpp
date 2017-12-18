#include"dog.h"
/*
计算出图片经lbp和GLCM后16维特征
参数：
a——图片a
返回值：
1*3的特征
*/
Mat threeColorFeature(Mat src)
{
	int rows = src.rows;
	int cols = src.cols;
	Mat imgHSV;
	vector<Mat> hsvSplit;
	cvtColor(src, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV
	split(imgHSV, hsvSplit);
	merge(hsvSplit, imgHSV);
	int hsv_h, hsv_s, hsv_v;
	int A = 0;
	double L = 0.0, L2 = 0.0, L3 = 0.0, L0;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			hsv_h = (hsvSplit[0].at<uchar>(i, j) * 255) / 180;
			hsv_s = hsvSplit[1].at<uchar>(i, j);
			hsv_v = hsvSplit[2].at<uchar>(i, j);
			if (hsv_h != 117)
			//if (hsv_h != 69)
			{
				L += hsv_h * 2 + hsv_v / 4 + hsv_v / 16;
				L += hsv_h;
				A++;
			}
		}
	}
	//int m = (hsvSplit[0].at<uchar>(5, 5) * 255) / 180;
	//cout <<  m<< endl;
	//cout << "A:" << A << endl;
	double u, v, s;
	u = L / A;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			hsv_h = (hsvSplit[0].at<uchar>(i, j) * 255) / 180;
			hsv_s = hsvSplit[1].at<uchar>(i, j);
			hsv_v = hsvSplit[2].at<uchar>(i, j);
			if (hsv_h != 37)
			{
				L0 = hsv_h * 2 + hsv_v / 4 + hsv_v / 16;
				L2 += (L0 - u) *(L0 - u);
				if (L0 >= u)
				{
					L3 += (L0 - u) *(L0 - u)*(L0 - u);
				}
				else
					L3 += (u - L0)*(u - L0)*(u - L0);
			}

		}
	}
	v = sqrt(L2 / A);
	s = pow(L3 / A, 1.0 / 3);
	Mat index = Mat(1, 3, CV_64FC1);
	index.at<Vec<double, 1>>(0, 0)[0] = u;
	index.at<Vec<double, 1>>(0, 1)[0] = v;
	index.at<Vec<double, 1>>(0, 2)[0] = s;
	return index;
}

void colorFeaturetrain(vector<picture> Tem)
{
	//grainFeatureInit(Tem);
	vector<picture> fea3(Tem.size());
	for (int i = 0; i < Tem.size(); i++) {
		fea3[i].pic = threeColorFeature(Tem[i].pic);
		fea3[i].name = Tem[i].name;
		cout << "颜色训练：" << i << endl;
	}
	writeFiles("color.xml", fea3);
}

vector<picture> colorFeaturematch(Mat src, vector<string> files, double th, int times)
{
	vector<picture> trained = readFiles("color.xml");
	vector<picture> select = getPicturesFromFiles(trained, files);
	Mat tmp = threeColorFeature(src);
	vector<picture> match = matchTemplate_colorFeature(tmp, select, th, times);
	return match;
}

