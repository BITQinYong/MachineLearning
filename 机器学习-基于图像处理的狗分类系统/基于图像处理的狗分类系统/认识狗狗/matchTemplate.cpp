#include<opencv2\core\core.hpp>
#include<opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2\xfeatures2d.hpp>
#include<math.h>
#include<iostream>
#include"dog.h"
using namespace std;
using namespace cv;
/*
	基于Hu不变矩的整体图像匹配
	调用了opencv的现有函数
	参数：
		src——源图片
		Tem[]——模板图
		numTem——模板数量
	返回值：
		picture
*/
picture matchTemplate_HuMatrix(Mat src, picture Tem[], int numTem)
{
	CV_Assert(src.channels() == 1);
	CV_Assert(src.depth() != sizeof(uchar));
	CV_Assert(numTem > 0);

	vector<double> Similarity(numTem);//src与每个模板的相似度

	for (int i = 0; i < numTem; i++) {
		/*计算hu矩阵*/
		double hu0[7], hu[7];
		Moments m0 = moments(src);
		Moments m = moments(Tem[i].pic);
		HuMoments(m0, hu0);
		HuMoments(m, hu);
		/*相似度检测*/
		double dbR = 0; //相似度
		double dSigmaST = 0;
		double dSigmaS = 0;
		double dSigmaT = 0;
		double temp = 0;
		for (int j = 0; j<7; j++)
		{
			temp = fabs(hu0[j] * hu[j]);
			dSigmaST += temp;
			dSigmaS += pow(hu0[j], 2);
			dSigmaT += pow(hu[j], 2);
		}
		//cout << dSigmaST << "  " << dSigmaS << "  " << dSigmaT<<endl;
		dbR = dSigmaST / (sqrt(dSigmaS)*sqrt(dSigmaT));
		Similarity[i] = dbR;
		cout << dbR << endl;
	}
	
	/*找出最相似的图片*/
	double max = 0;
	int pos = 0;
	for (int i = 0; i < Similarity.size(); i++) {
		if (Similarity[i] > max) {
			max = Similarity[i];
			pos = i;
		}
	}

	/*返回*/
	picture pic;
	pic.name = Tem[pos].name;
	pic.pic = src;
	return pic;
}

/*
	基于sift和RANSAC进行匹配，根据匹配点的数目进行选择,直接匹配
	调用了opencv的现有函数
	参数：
		src——源图片sift特征
		Tem[]——模板图
	返回值：
		picture
*/
picture matchTemplate_SiftRANSAC(Mat src, vector<picture> Tem)
{
	int numTem = Tem.size();

	CV_Assert(src.depth() != sizeof(uchar));
	CV_Assert(numTem > 0);

	vector<int> numMatchPoint(numTem);
	for (int k = 0; k < numTem; k++) {
		/*sift特征提取*/
		//xfeatures2d::SiftFeatureDetector detector;
		Ptr<Feature2D> sift= xfeatures2d::SIFT::create();
		vector<KeyPoint> keypoint, keypoint0;//特征点
		sift->detect(src, keypoint);
		sift->detect(Tem[k].pic, keypoint0);
		/*提取特征点的特征向量（128维）*/
		xfeatures2d::SiftDescriptorExtractor extractor;
		Mat descriptor, descriptor0;//特征向量
		sift->compute(src, keypoint, descriptor);
		sift->compute(Tem[k].pic, keypoint0, descriptor0);
		/*匹配特征点，主要计算两个特征点特征向量的欧式距离*/
		BFMatcher matcher;//暴力匹配
		vector<DMatch> matches;
		matcher.match(descriptor, descriptor0, matches);

		/*
			RANSAC 消除误匹配特征点
				1）根据matches将特征点对齐,将坐标转换为float类型
				2）使用求基础矩阵方法 findFundamentalMat,得到RansacStatus
				3）根据RansacStatus来将误匹配的点也即RansacStatus[i]=0的点删除
		*/
		//步骤一：特征点对齐,坐标转换为float型
		vector<KeyPoint> R_keypoint, R_keypoint0;
		for (size_t i = 0; i<matches.size(); i++)
		{
			R_keypoint.push_back(keypoint[matches[i].queryIdx]);//R_keypoint1提取src中能与模板Tem[i]匹配的特征点，
			R_keypoint0.push_back(keypoint0[matches[i].trainIdx]);//matches中存储了这些匹配点对的src和Tem[i]的索引值
		}
		vector<Point2f>float_p, float_p0;//坐标转换
		for (size_t i = 0; i<matches.size(); i++)
		{
			float_p.push_back(R_keypoint[i].pt);//pt为整形坐标
			float_p0.push_back(R_keypoint0[i].pt);
		}

		//步骤二：利用基础矩阵得到RansacStatus（状态）
		vector<uchar> RansacStatus;
		Mat Fundamental = findFundamentalMat(float_p, float_p0, RansacStatus, FM_RANSAC);

		//步骤三：根据RansacStatus来将误匹配的点也即RansacStatus[i]=0的点删除
		vector<KeyPoint> RR_keypoint, RR_keypoint0;
		vector<DMatch> RR_matches;            //重新定义RR_keypoint 和RR_matches来存储新的关键点和匹配矩阵
		int index = 0;
		for (size_t i = 0; i < matches.size(); i++)
		{
			if (RansacStatus[i] != 0)
			{
				RR_keypoint.push_back(R_keypoint[i]);
				RR_keypoint0.push_back(R_keypoint0[i]);
				matches[i].queryIdx = index;
				matches[i].trainIdx = index;
				RR_matches.push_back(matches[i]);
				index++;
			}
		}

		/*存储匹配点个数*/
		numMatchPoint[k] = RR_matches.size();
	}

	/*寻找匹配点最多的图片*/
	int max = 0;
	int pos = 0;
	for (int i = 0; i < numTem; i++) {
		if (numMatchPoint[i] > max) {
			max = numMatchPoint[i];
			pos = i;
		}
	}

	/*返回picture*/
	picture pic;
	pic.pic = src;
	pic.name = Tem[pos].name;
	return pic;
}

/*
	基于sift和RANSAC进行匹配，根据匹配点的数目进行选择,利用存储点匹配
	调用了opencv的现有函数
	参数：
		src——源图片
		src_des——源图片sift向量
		src_KP——源图片sift特征点
		Tem——模板图片sift向量
		Tem_KP——模板图片sift特征点
	返回值：
		picture
*/
picture matchTemplate_SiftRANSAC_SAVE(Mat src, Mat src_des, vector<KeyPoint> src_KP, vector<picture> Tem, vector<vector<KeyPoint>> Tem_KP)
{
	int numTem = Tem.size();

	CV_Assert(numTem > 0);

	vector<int> numMatchPoint(numTem);
	for (int k = 0; k < numTem; k++) {
		/*匹配特征点，主要计算两个特征点特征向量的欧式距离*/
		BFMatcher matcher;//暴力匹配
		vector<DMatch> matches;
		matcher.match(src_des, Tem[k].pic, matches);

		/*
			RANSAC 消除误匹配特征点
				1）根据matches将特征点对齐,将坐标转换为float类型
				2）使用求基础矩阵方法 findFundamentalMat,得到RansacStatus
				3）根据RansacStatus来将误匹配的点也即RansacStatus[i]=0的点删除
		*/
		//步骤一：特征点对齐,坐标转换为float型
		vector<KeyPoint> R_keypoint, R_keypoint0;
		for (size_t i = 0; i<matches.size(); i++)
		{
			R_keypoint.push_back(src_KP[matches[i].queryIdx]);//R_keypoint1提取src中能与模板Tem[i]匹配的特征点，
			R_keypoint0.push_back(Tem_KP[k][matches[i].trainIdx]);//matches中存储了这些匹配点对的src和Tem[i]的索引值
		}
		vector<Point2f>float_p, float_p0;//坐标转换
		for (size_t i = 0; i<matches.size(); i++)
		{
			float_p.push_back(R_keypoint[i].pt);//pt为整形坐标
			float_p0.push_back(R_keypoint0[i].pt);
		}

		//步骤二：利用基础矩阵得到RansacStatus（状态）
		vector<uchar> RansacStatus;
		Mat Fundamental = findFundamentalMat(float_p, float_p0, RansacStatus, FM_RANSAC);

		//步骤三：根据RansacStatus来将误匹配的点也即RansacStatus[i]=0的点删除
		vector<KeyPoint> RR_keypoint, RR_keypoint0;
		vector<DMatch> RR_matches;            //重新定义RR_keypoint 和RR_matches来存储新的关键点和匹配矩阵
		int index = 0;
		for (size_t i = 0; i < matches.size(); i++)
		{
			if (RansacStatus[i] != 0)
			{
				RR_keypoint.push_back(R_keypoint[i]);
				RR_keypoint0.push_back(R_keypoint0[i]);
				matches[i].queryIdx = index;
				matches[i].trainIdx = index;
				RR_matches.push_back(matches[i]);
				index++;
			}
		}

		/*存储匹配点个数*/
		numMatchPoint[k] = RR_matches.size();
	}

	/*寻找匹配点最多的图片*/
	int max = 0;
	int pos = 0;
	for (int i = 0; i < numTem; i++) {
		if (numMatchPoint[i] > max) {
			max = numMatchPoint[i];
			pos = i;
		}
	}

	/*返回picture*/
	picture pic;
	pic.pic = src;
	pic.name = Tem[pos].name;
	return pic;
}

/*
	基于sift和劳式算法进行匹配，根据匹配点的数目进行选择,利用存储点匹配
	调用了opencv的现有函数
	参数：
		src——源图片
		src_des——源图片sift向量
		src_KP——源图片sift特征点
		Tem——模板图片sift向量
		Tem_KP——模板图片sift特征点
	返回值：
		picture
*/
picture matchTemplate_SiftLowe_SAVE(Mat src, Mat src_des, vector<KeyPoint> src_KP, vector<picture> Tem, vector<vector<KeyPoint>> Tem_KP)
{
	int numTem = Tem.size();

	CV_Assert(numTem > 0);

	/*匹配特征点，主要计算两个特征点特征向量的欧式距离*/
	BFMatcher matcher;//暴力匹配
	vector<Mat> srcTrain(1, src_des);
	matcher.add(srcTrain);
	matcher.train();

	vector<int> numMatchPoint(numTem);
	for (int k = 0; k < numTem; k++) {

		vector<vector<DMatch>> matches;
		matcher.knnMatch(Tem[k].pic, matches, 2);

		//Mat img_matches;
		//drawMatches(src, src_KP, Tem[k].pic, Tem_KP[k], matches, img_matches);
		//imshow("误匹配消除前", img_matches);

		/*
		根据劳式算法得到优秀的匹配点
		*/
		vector<DMatch> goodMatches;
		for (unsigned int i = 0; i < matches.size(); i++) {
			if (matches[i][0].distance < 0.6*matches[i][1].distance) {
				goodMatches.push_back(matches[i][0]);
			}
		}

		//Mat img_matches1;
		//drawMatches(src, src_KP, Tem[k].pic, Tem_KP[k], goodMatches, img_matches1);
		//imshow("误匹配消除后", img_matches1);
		//waitKey(0);

		/*存储匹配点个数*/
		numMatchPoint[k] = goodMatches.size();
	}

	/*寻找匹配点最多的图片*/
	int max = 0;
	int pos = 0;
	for (int i = 0; i < numTem; i++) {
		if (numMatchPoint[i] > max) {
			max = numMatchPoint[i];
			pos = i;
		}
	}

	/*返回picture*/
	picture pic;
	pic.pic = src;
	pic.name = Tem[pos].name;
	return pic;
}


struct Mdistance
{
	double d;//马氏距离
	int which;//索引
	bool operator <(const Mdistance& rhs)const   //升序排序时必须写的函数
	{
		return   d < rhs.d;
	}
};
double variance(vector<Mdistance> v)
{
	double sum = 0;
	for (int i = 0; i < v.size(); i++) {
		sum += v[i].d;
	}
	double mean = sum / v.size();
	double var = 0;
	for (int i = 0; i < v.size(); i++) {
		var += (v[i].d - mean)*(v[i].d - mean);
	}
	return var;
}
/*
	基于LBP和GLCM进行匹配，根据向量的马氏距离进行选择,进行筛选
	调用了opencv的现有函数
	参数：
		src——源图片
		Tem[]——模板图，每个必须是16维的特征点
		numTem——模板数量
		th——阈值[0,1],阈值越大，筛选的粒度会越细，即删除的会越多
		times——迭代次数
	返回值：
		picture
*/
vector<picture> matchTemplate_grainFeature(Mat src, vector<picture> Tem, double th, int times)
{
	for (int p = 0; p < times; p++) {
		int numTem = Tem.size();
		CV_Assert(src.channels() == 1);
		CV_Assert(src.depth() != sizeof(double));
		CV_Assert(src.rows == 1 && src.cols == 16);
		for (int i = 0; i < numTem; i++) {
			CV_Assert(Tem[i].pic.rows == 1 && Tem[i].pic.cols == 16);
		}

		vector<Mdistance> Mdis(numTem);
		for (int i = 0; i < numTem; i++) {//计算马氏距离
			Mdis[i].d = Mahalanobis(src, Tem[i].pic);
			Mdis[i].which = i;
		}
		sort(Mdis.begin(), Mdis.end());//升序排序
		vector<Mdistance> d0, d1, d2;
		for (int i = 0; i < numTem; i++) {
			double tmp = Mahalanobis(Tem[Mdis[i].which].pic, Tem[Mdis[numTem - 1].which].pic);
			d0.push_back(Mdis[i]);
			if (Mdis[i].d <= tmp) {
				d1.push_back(Mdis[i]);
			}
			else {
				d2.push_back(Mdis[i]);
			}
		}
		vector<Mdistance> d = ((variance(d1) + variance(d2)) / variance(d0)) >= th ? d0 : d1;
		vector<picture> vecPic(d.size());
		for (int i = 0; i < d.size(); i++) {
			vecPic[i] = Tem[d[i].which];
		}
		Tem = vecPic;
	}
	
	return Tem;
}


struct Cdistance
{
	double d;//距离
	int which;//索引
	bool operator <(const Cdistance& rhs)const   //升序排序时必须写的函数
	{
		return   d < rhs.d;
	}
};
double Cvariance(vector<Cdistance> v)
{
	double sum = 0;
	for (int i = 0; i < v.size(); i++) {
		sum += v[i].d;
	}
	double mean = sum / v.size();
	double var = 0;
	for (int i = 0; i < v.size(); i++) {
		var += (v[i].d - mean)*(v[i].d - mean);
	}
	return var;
}

vector<picture> matchTemplate_colorFeature(Mat src, vector<picture> Tem, double th, int times)
{
	for (int p = 0; p < times; p++) {
		int numTem = Tem.size();
		CV_Assert(src.channels() == 1);
		CV_Assert(src.depth() != sizeof(double));
		CV_Assert(src.rows == 1 && src.cols == 3);
		for (int i = 0; i < numTem; i++) {
			CV_Assert(Tem[i].pic.rows == 1 && Tem[i].pic.cols == 3);
		}

		vector<Cdistance> Cdis(numTem);
		for (int i = 0; i < numTem; i++) {//计算距离
			Cdis[i].d = colordis(src, Tem[i].pic);
			Cdis[i].which = i;
		}
		sort(Cdis.begin(), Cdis.end());//升序排序
		vector<Cdistance> d0, d1, d2;
		for (int i = 0; i < numTem; i++) {
			double tmp = colordis(Tem[Cdis[i].which].pic, Tem[Cdis[numTem - 1].which].pic);
			d0.push_back(Cdis[i]);
			if (Cdis[i].d <= tmp) {
				d1.push_back(Cdis[i]);
			}
			else {
				d2.push_back(Cdis[i]);
			}
		}
		vector<Cdistance> d = ((Cvariance(d1) + Cvariance(d2)) / Cvariance(d0)) >= th ? d0 : d1;
		vector<picture> vecPic(d.size());
		for (int i = 0; i < d.size(); i++) {
			vecPic[i] = Tem[d[i].which];
		}
		Tem = vecPic;
	}
	
	return Tem;
}




struct Sdistance
{
	double d;//距离
	int which;//索引
	bool operator <(const Sdistance& rhs)const   //升序排序时必须写的函数
	{
		return   d < rhs.d;
	}
};
double Svariance(vector<Sdistance> v)
{
	double sum = 0;
	for (int i = 0; i < v.size(); i++) {
		sum += v[i].d;
	}
	double mean = sum / v.size();
	double var = 0;
	for (int i = 0; i < v.size(); i++) {
		var += (v[i].d - mean)*(v[i].d - mean);
	}
	return var;
}

vector<picture> matchTemplate_shapeFeature(Mat src, vector<picture> Tem, double th, int times)
{
	for (int p = 0; p < times; p++) {
		int numTem = Tem.size();
		CV_Assert(src.channels() == 1);
		CV_Assert(src.depth() != sizeof(double));
		CV_Assert(src.rows >0 && src.cols>0);
		for (int i = 0; i < numTem; i++) {
			CV_Assert(Tem[i].pic.rows >0 && Tem[i].pic.cols >0);
		}

		vector<Sdistance> Sdis(numTem);
		for (int i = 0; i < numTem; i++) {//计算距离
			Sdis[i].d = shapedis(src, Tem[i].pic);
			Sdis[i].which = i;
		}
		sort(Sdis.begin(), Sdis.end());//升序排序
		vector<Sdistance> d0, d1, d2;
		for (int i = 0; i < numTem; i++) {
			double tmp = shapedis(Tem[Sdis[i].which].pic, Tem[Sdis[numTem - 1].which].pic);
			d0.push_back(Sdis[i]);
			if (Sdis[i].d <= tmp) {
				d1.push_back(Sdis[i]);
			}
			else {
				d2.push_back(Sdis[i]);
			}
		}
		vector<Sdistance> d = ((Svariance(d1) + Svariance(d2)) / Svariance(d0)) >= th ? d0 : d1;
		vector<picture> vecPic(d.size());
		for (int i = 0; i < d.size(); i++) {
			vecPic[i] = Tem[d[i].which];
		}
		Tem = vecPic;
	}

	return Tem;
}
