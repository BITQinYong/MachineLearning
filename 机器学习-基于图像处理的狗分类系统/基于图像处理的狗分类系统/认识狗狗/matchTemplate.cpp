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
	����Hu����ص�����ͼ��ƥ��
	������opencv�����к���
	������
		src����ԴͼƬ
		Tem[]����ģ��ͼ
		numTem����ģ������
	����ֵ��
		picture
*/
picture matchTemplate_HuMatrix(Mat src, picture Tem[], int numTem)
{
	CV_Assert(src.channels() == 1);
	CV_Assert(src.depth() != sizeof(uchar));
	CV_Assert(numTem > 0);

	vector<double> Similarity(numTem);//src��ÿ��ģ������ƶ�

	for (int i = 0; i < numTem; i++) {
		/*����hu����*/
		double hu0[7], hu[7];
		Moments m0 = moments(src);
		Moments m = moments(Tem[i].pic);
		HuMoments(m0, hu0);
		HuMoments(m, hu);
		/*���ƶȼ��*/
		double dbR = 0; //���ƶ�
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
	
	/*�ҳ������Ƶ�ͼƬ*/
	double max = 0;
	int pos = 0;
	for (int i = 0; i < Similarity.size(); i++) {
		if (Similarity[i] > max) {
			max = Similarity[i];
			pos = i;
		}
	}

	/*����*/
	picture pic;
	pic.name = Tem[pos].name;
	pic.pic = src;
	return pic;
}

/*
	����sift��RANSAC����ƥ�䣬����ƥ������Ŀ����ѡ��,ֱ��ƥ��
	������opencv�����к���
	������
		src����ԴͼƬsift����
		Tem[]����ģ��ͼ
	����ֵ��
		picture
*/
picture matchTemplate_SiftRANSAC(Mat src, vector<picture> Tem)
{
	int numTem = Tem.size();

	CV_Assert(src.depth() != sizeof(uchar));
	CV_Assert(numTem > 0);

	vector<int> numMatchPoint(numTem);
	for (int k = 0; k < numTem; k++) {
		/*sift������ȡ*/
		//xfeatures2d::SiftFeatureDetector detector;
		Ptr<Feature2D> sift= xfeatures2d::SIFT::create();
		vector<KeyPoint> keypoint, keypoint0;//������
		sift->detect(src, keypoint);
		sift->detect(Tem[k].pic, keypoint0);
		/*��ȡ�����������������128ά��*/
		xfeatures2d::SiftDescriptorExtractor extractor;
		Mat descriptor, descriptor0;//��������
		sift->compute(src, keypoint, descriptor);
		sift->compute(Tem[k].pic, keypoint0, descriptor0);
		/*ƥ�������㣬��Ҫ������������������������ŷʽ����*/
		BFMatcher matcher;//����ƥ��
		vector<DMatch> matches;
		matcher.match(descriptor, descriptor0, matches);

		/*
			RANSAC ������ƥ��������
				1������matches�����������,������ת��Ϊfloat����
				2��ʹ����������󷽷� findFundamentalMat,�õ�RansacStatus
				3������RansacStatus������ƥ��ĵ�Ҳ��RansacStatus[i]=0�ĵ�ɾ��
		*/
		//����һ�����������,����ת��Ϊfloat��
		vector<KeyPoint> R_keypoint, R_keypoint0;
		for (size_t i = 0; i<matches.size(); i++)
		{
			R_keypoint.push_back(keypoint[matches[i].queryIdx]);//R_keypoint1��ȡsrc������ģ��Tem[i]ƥ��������㣬
			R_keypoint0.push_back(keypoint0[matches[i].trainIdx]);//matches�д洢����Щƥ���Ե�src��Tem[i]������ֵ
		}
		vector<Point2f>float_p, float_p0;//����ת��
		for (size_t i = 0; i<matches.size(); i++)
		{
			float_p.push_back(R_keypoint[i].pt);//ptΪ��������
			float_p0.push_back(R_keypoint0[i].pt);
		}

		//����������û�������õ�RansacStatus��״̬��
		vector<uchar> RansacStatus;
		Mat Fundamental = findFundamentalMat(float_p, float_p0, RansacStatus, FM_RANSAC);

		//������������RansacStatus������ƥ��ĵ�Ҳ��RansacStatus[i]=0�ĵ�ɾ��
		vector<KeyPoint> RR_keypoint, RR_keypoint0;
		vector<DMatch> RR_matches;            //���¶���RR_keypoint ��RR_matches���洢�µĹؼ����ƥ�����
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

		/*�洢ƥ������*/
		numMatchPoint[k] = RR_matches.size();
	}

	/*Ѱ��ƥ�������ͼƬ*/
	int max = 0;
	int pos = 0;
	for (int i = 0; i < numTem; i++) {
		if (numMatchPoint[i] > max) {
			max = numMatchPoint[i];
			pos = i;
		}
	}

	/*����picture*/
	picture pic;
	pic.pic = src;
	pic.name = Tem[pos].name;
	return pic;
}

/*
	����sift��RANSAC����ƥ�䣬����ƥ������Ŀ����ѡ��,���ô洢��ƥ��
	������opencv�����к���
	������
		src����ԴͼƬ
		src_des����ԴͼƬsift����
		src_KP����ԴͼƬsift������
		Tem����ģ��ͼƬsift����
		Tem_KP����ģ��ͼƬsift������
	����ֵ��
		picture
*/
picture matchTemplate_SiftRANSAC_SAVE(Mat src, Mat src_des, vector<KeyPoint> src_KP, vector<picture> Tem, vector<vector<KeyPoint>> Tem_KP)
{
	int numTem = Tem.size();

	CV_Assert(numTem > 0);

	vector<int> numMatchPoint(numTem);
	for (int k = 0; k < numTem; k++) {
		/*ƥ�������㣬��Ҫ������������������������ŷʽ����*/
		BFMatcher matcher;//����ƥ��
		vector<DMatch> matches;
		matcher.match(src_des, Tem[k].pic, matches);

		/*
			RANSAC ������ƥ��������
				1������matches�����������,������ת��Ϊfloat����
				2��ʹ����������󷽷� findFundamentalMat,�õ�RansacStatus
				3������RansacStatus������ƥ��ĵ�Ҳ��RansacStatus[i]=0�ĵ�ɾ��
		*/
		//����һ�����������,����ת��Ϊfloat��
		vector<KeyPoint> R_keypoint, R_keypoint0;
		for (size_t i = 0; i<matches.size(); i++)
		{
			R_keypoint.push_back(src_KP[matches[i].queryIdx]);//R_keypoint1��ȡsrc������ģ��Tem[i]ƥ��������㣬
			R_keypoint0.push_back(Tem_KP[k][matches[i].trainIdx]);//matches�д洢����Щƥ���Ե�src��Tem[i]������ֵ
		}
		vector<Point2f>float_p, float_p0;//����ת��
		for (size_t i = 0; i<matches.size(); i++)
		{
			float_p.push_back(R_keypoint[i].pt);//ptΪ��������
			float_p0.push_back(R_keypoint0[i].pt);
		}

		//����������û�������õ�RansacStatus��״̬��
		vector<uchar> RansacStatus;
		Mat Fundamental = findFundamentalMat(float_p, float_p0, RansacStatus, FM_RANSAC);

		//������������RansacStatus������ƥ��ĵ�Ҳ��RansacStatus[i]=0�ĵ�ɾ��
		vector<KeyPoint> RR_keypoint, RR_keypoint0;
		vector<DMatch> RR_matches;            //���¶���RR_keypoint ��RR_matches���洢�µĹؼ����ƥ�����
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

		/*�洢ƥ������*/
		numMatchPoint[k] = RR_matches.size();
	}

	/*Ѱ��ƥ�������ͼƬ*/
	int max = 0;
	int pos = 0;
	for (int i = 0; i < numTem; i++) {
		if (numMatchPoint[i] > max) {
			max = numMatchPoint[i];
			pos = i;
		}
	}

	/*����picture*/
	picture pic;
	pic.pic = src;
	pic.name = Tem[pos].name;
	return pic;
}

/*
	����sift����ʽ�㷨����ƥ�䣬����ƥ������Ŀ����ѡ��,���ô洢��ƥ��
	������opencv�����к���
	������
		src����ԴͼƬ
		src_des����ԴͼƬsift����
		src_KP����ԴͼƬsift������
		Tem����ģ��ͼƬsift����
		Tem_KP����ģ��ͼƬsift������
	����ֵ��
		picture
*/
picture matchTemplate_SiftLowe_SAVE(Mat src, Mat src_des, vector<KeyPoint> src_KP, vector<picture> Tem, vector<vector<KeyPoint>> Tem_KP)
{
	int numTem = Tem.size();

	CV_Assert(numTem > 0);

	/*ƥ�������㣬��Ҫ������������������������ŷʽ����*/
	BFMatcher matcher;//����ƥ��
	vector<Mat> srcTrain(1, src_des);
	matcher.add(srcTrain);
	matcher.train();

	vector<int> numMatchPoint(numTem);
	for (int k = 0; k < numTem; k++) {

		vector<vector<DMatch>> matches;
		matcher.knnMatch(Tem[k].pic, matches, 2);

		//Mat img_matches;
		//drawMatches(src, src_KP, Tem[k].pic, Tem_KP[k], matches, img_matches);
		//imshow("��ƥ������ǰ", img_matches);

		/*
		������ʽ�㷨�õ������ƥ���
		*/
		vector<DMatch> goodMatches;
		for (unsigned int i = 0; i < matches.size(); i++) {
			if (matches[i][0].distance < 0.6*matches[i][1].distance) {
				goodMatches.push_back(matches[i][0]);
			}
		}

		//Mat img_matches1;
		//drawMatches(src, src_KP, Tem[k].pic, Tem_KP[k], goodMatches, img_matches1);
		//imshow("��ƥ��������", img_matches1);
		//waitKey(0);

		/*�洢ƥ������*/
		numMatchPoint[k] = goodMatches.size();
	}

	/*Ѱ��ƥ�������ͼƬ*/
	int max = 0;
	int pos = 0;
	for (int i = 0; i < numTem; i++) {
		if (numMatchPoint[i] > max) {
			max = numMatchPoint[i];
			pos = i;
		}
	}

	/*����picture*/
	picture pic;
	pic.pic = src;
	pic.name = Tem[pos].name;
	return pic;
}


struct Mdistance
{
	double d;//���Ͼ���
	int which;//����
	bool operator <(const Mdistance& rhs)const   //��������ʱ����д�ĺ���
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
	����LBP��GLCM����ƥ�䣬�������������Ͼ������ѡ��,����ɸѡ
	������opencv�����к���
	������
		src����ԴͼƬ
		Tem[]����ģ��ͼ��ÿ��������16ά��������
		numTem����ģ������
		th������ֵ[0,1],��ֵԽ��ɸѡ�����Ȼ�Խϸ����ɾ���Ļ�Խ��
		times������������
	����ֵ��
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
		for (int i = 0; i < numTem; i++) {//�������Ͼ���
			Mdis[i].d = Mahalanobis(src, Tem[i].pic);
			Mdis[i].which = i;
		}
		sort(Mdis.begin(), Mdis.end());//��������
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
	double d;//����
	int which;//����
	bool operator <(const Cdistance& rhs)const   //��������ʱ����д�ĺ���
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
		for (int i = 0; i < numTem; i++) {//�������
			Cdis[i].d = colordis(src, Tem[i].pic);
			Cdis[i].which = i;
		}
		sort(Cdis.begin(), Cdis.end());//��������
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
	double d;//����
	int which;//����
	bool operator <(const Sdistance& rhs)const   //��������ʱ����д�ĺ���
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
		for (int i = 0; i < numTem; i++) {//�������
			Sdis[i].d = shapedis(src, Tem[i].pic);
			Sdis[i].which = i;
		}
		sort(Sdis.begin(), Sdis.end());//��������
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
