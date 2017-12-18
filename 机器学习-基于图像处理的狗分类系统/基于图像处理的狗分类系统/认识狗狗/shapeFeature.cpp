#include"dog.h"

Mat shapeFeature(Mat src)
{
	int rows = src.rows;
	int cols = src.cols;
	Mat hecopy;
	hecopy = src.clone();
	//-------------------------------二值化处理----------------
	int kw1, kw2, kw3;
	for (int i = 0; i < hecopy.rows; i++) {
		uchar *date = hecopy.ptr(i);
		for (int j = 0; j < hecopy.cols; j++) {
			kw1 = date[j * 3 + 0];
			kw2 = date[j * 3 + 1];
			kw3 = date[j * 3 + 2];
			//cout << kw1 << " " << kw2 << " " << kw3<<endl;
			//if (date[j * 3 + 0] <20 && date[j * 3 + 1] >205 && date[j * 3 + 2] <20 )
			if (kw1 >160 && kw1<200 && kw2 >207 && kw2<247 && kw3 <47 && kw3>7)
			//if (kw1 >53 && kw1<93 && kw2 >170 && kw2<210 && kw3 <134 && kw3>94)
			{
				date[j * 3 + 0] = date[j * 3 + 1] = date[j * 3 + 2] = 0;
			}
			else {
				date[j * 3 + 0] = date[j * 3 + 1] = date[j * 3 + 2] = 255;
			}
		}
	}
	//imshow("hecopy", hecopy);
	//cvWaitKey(0);

	//------------------------------均值滤波处理-------------------------
	Mat src2;
	cvtColor(hecopy, src2, CV_BGR2GRAY);
	//imshow("灰度", src);
	//int hh = src.at<uchar>(3, 3);
	//cout << hh << endl;
	//for (int i = 0; i < hecopy.rows; i++) {
	//	for (int j = 0; j < hecopy.cols; j++) {
	//		if (src.at<uchar>(i, j) == 162)
	//			src.at<uchar>(i, j) = 255;
	//		else src.at<uchar>(i, j) = 0;
	//		}
	//	}

	//int m = src.at<uchar>(6, 10);
	//cout << m<< endl;
	//imshow("src", src);
	Mat dst;
	//cvtColor(hecopy, dst, CV_BGR2GRAY);
	dst.create(rows, cols, src2.type());   //创建输出图像
	int k0[3] = { 0 }, k1[3] = { 0 }, k[3] = { 0 }, hehe = 3;
	int Row[3], Col[3];
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			Row[0] = i - 1;
			Col[0] = j - 1;
			Row[1] = i;
			Col[1] = j;
			Row[2] = i + 1;
			Col[2] = j + 1;
			if (Row[0]<0) Row[0] = 0;//判断边界
			if (Col[0]<0) Col[0] = 0;
			if (Row[2] >= rows)
				Row[2] = rows - 1;
			if (Col[2] >= cols)
				Col[2] = cols - 1;
			dst.at<uchar>(i, j) = (src2.at<uchar>(Row[0], Col[1]) + src2.at<uchar>(Row[1], Col[0]) + src2.at<uchar>(Row[2], Col[1]) + src2.at<uchar>(Row[1], Col[2])
				+ src2.at<uchar>(Row[0], Col[0]) + src2.at<uchar>(Row[2], Col[0]) + src2.at<uchar>(Row[2], Col[2]) + src2.at<uchar>(Row[0], Col[2])
				+ src2.at<uchar>(Row[1], Col[1])) / 9;
		}
	}
	//imshow("滤波后", dst);
	//----------------------------------提取图像的边界轮廓
	GaussianBlur(dst, dst, Size(3, 3), 0);
	//Canny(dst, dst, 100, 250);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (dst.at<uchar>(i, j) <100&& dst.at<uchar>(i, j) >30)
				dst.at<uchar>(i, j) = 255;
			else dst.at<uchar>(i, j) = 0;
		}
	}

	//imshow("图像轮廓", dst);

	//vector<vector<Point>> w;
	////vector<Vec4i> hehe;
	//vector<Vec4i> hierarchy;
	////vector<vector<KeyPoint>>   hehe;
	////std::vector> contours;
	//findContours(dst,w , hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point());//提取轮廓元素
	return dst;
	////w[0] = 1;
	//origin.shape_contours = w;
	//Mat imageContours = Mat::zeros(dst.size(), CV_8UC1);
	//Mat Contours = Mat::zeros(dst.size(), CV_8UC1);  //绘制  
	//for (int i = 0; i<w.size(); i++)
	//{
	//	//contours[i]代表的是第i个轮廓，contours[i].size()代表的是第i个轮廓上所有的像素点数  
	//	for (int j = 0; j<w[i].size(); j++)
	//	{
	//		//绘制出contours向量内所有的像素点  
	//		Point P = Point(w[i][j].x, w[i][j].y);
	//		Contours.at<uchar>(P) = 255;
	//	}
	//	//绘制轮廓  
	//	drawContours(imageContours, w, i, Scalar(255), 1, 8, hierarchy);
	//}
	//imshow("Contours Image", imageContours); //轮廓  
	//imshow("Point of Contours", Contours);   //向量contours内保存的所有轮廓点集 
	//cvMatchShapes(dst, dst, 1, 0);
}

void shapeFeaturetrain(vector<picture> Tem)
{
	vector<picture> feashape(Tem.size());
	for (int i = 0; i < Tem.size(); i++) {
		feashape[i].pic = shapeFeature(Tem[i].pic);
		feashape[i].name = Tem[i].name;
		cout << "形状训练：" << i << endl;
	}
	writeFiles("shape.xml", feashape);
}

vector<picture> shapeFeaturematch(Mat src, vector<string> files, double th, int times)
{
	vector<picture> trained = readFiles("shape.xml");
	vector<picture> select = getPicturesFromFiles(trained, files);
	Mat tmp = shapeFeature(src);
	vector<picture> match = matchTemplate_shapeFeature(tmp, select, th, times);
	return match;
}