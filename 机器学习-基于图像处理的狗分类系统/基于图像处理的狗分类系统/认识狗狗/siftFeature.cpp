#include"dog.h"
using namespace std;
using namespace cv;
/*
�����ͼƬsift128ά����,������������
������
src����ͼƬsrc
descriptor������������
vector<KeyPoint>����������
����ֵ��
��
*/
void siftFeature128(Mat src, Mat& descriptor, vector<KeyPoint>& keypoint)
{
	/*sift������ȡ */
	//xfeatures2d::SiftFeatureDetector detector;
	Ptr<Feature2D> sift = xfeatures2d::SIFT::create();
	/*������*/
	sift->detect(src, keypoint);
	/*��ȡ�����������������128ά��*/
	xfeatures2d::SiftDescriptorExtractor extractor;
	/*��������*/
	sift->compute(src, keypoint, descriptor);
}

/*
������ͼƬ���г�ʼ��
������
Tem����ͼƬ��
����ֵ��
��
*/
void siftFeatureInit(vector<picture>& Tem)
{
	for (int i = 0; i < Tem.size(); i++) {
		blur(Tem[i].pic, Tem[i].pic, Size(3, 3));
	}
}

/*
������ͼƬ����ѵ������¼��sift�ص�
������
Tem����ͼƬ��
����ֵ��
��
*/
void siftFeaturetrain(vector<picture> Tem)
{
	//siftFeatureInit(Tem);
	vector<picture> fea128(Tem.size());
	vector<vector<KeyPoint>> KP(Tem.size());
	for (int i = 0; i < Tem.size(); i++) {
		siftFeature128(Tem[i].pic, fea128[i].pic, KP[i]);
		fea128[i].name = Tem[i].name;
		cout << "siftѵ��:" << i << endl;
	}
	writeFiles("sift128Vector.xml", fea128);
	writeFiles_KeyPoint("siftKeyPoint.xml", KP);
}

/*
��ͼƬ��Χ����ɸѡ
������
src����ԴͼƬ
����ֵ��
ͼƬ��
*/
picture siftFeaturematch(Mat src, vector<string> files)
{
	//��ȡԴͼƬ����
	vector<KeyPoint> src_KP;
	Mat src_des;
	siftFeature128(src, src_des, src_KP);

	//��ȡѵ��ͼƬ����
	vector<vector<KeyPoint>> train_KP0 = readFiles_KeyPoint("siftKeyPoint.xml");
	vector<picture> trained0 = readFiles("sift128Vector.xml");

	//ѡ����Ҫƥ�������
	vector<picture> select = getPicturesFromFiles(trained0, files);
	vector<vector<KeyPoint>> train_KP;
	vector<picture> trained;
	for (int i = 0; i < trained0.size(); i++) {
		for (int j = 0; j < select.size(); j++) {
			if (trained0[i].name == select[j].name) {
				trained.push_back(trained0[i]);
				train_KP.push_back(train_KP0[i]);
			}
		}
	}

	//picture match = matchTemplate_SiftRANSAC_SAVE(src, src_des, src_KP, trained, train_KP);
	picture match = matchTemplate_SiftLowe_SAVE(src, src_des, src_KP, trained, train_KP);
	return match;
}