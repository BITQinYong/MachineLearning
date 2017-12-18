#include"dog.h"
using namespace std;
using namespace cv;
/*
计算出图片sift128维特征,保留其特征点
参数：
src——图片src
descriptor——特征向量
vector<KeyPoint>——特征点
返回值：
无
*/
void siftFeature128(Mat src, Mat& descriptor, vector<KeyPoint>& keypoint)
{
	/*sift特征提取 */
	//xfeatures2d::SiftFeatureDetector detector;
	Ptr<Feature2D> sift = xfeatures2d::SIFT::create();
	/*特征点*/
	sift->detect(src, keypoint);
	/*提取特征点的特征向量（128维）*/
	xfeatures2d::SiftDescriptorExtractor extractor;
	/*特征向量*/
	sift->compute(src, keypoint, descriptor);
}

/*
对已有图片进行初始化
参数：
Tem——图片集
返回值：
无
*/
void siftFeatureInit(vector<picture>& Tem)
{
	for (int i = 0; i < Tem.size(); i++) {
		blur(Tem[i].pic, Tem[i].pic, Size(3, 3));
	}
}

/*
对已有图片进行训练，记录其sift特点
参数：
Tem——图片集
返回值：
无
*/
void siftFeaturetrain(vector<picture> Tem)
{
	//siftFeatureInit(Tem);
	vector<picture> fea128(Tem.size());
	vector<vector<KeyPoint>> KP(Tem.size());
	for (int i = 0; i < Tem.size(); i++) {
		siftFeature128(Tem[i].pic, fea128[i].pic, KP[i]);
		fea128[i].name = Tem[i].name;
		cout << "sift训练:" << i << endl;
	}
	writeFiles("sift128Vector.xml", fea128);
	writeFiles_KeyPoint("siftKeyPoint.xml", KP);
}

/*
对图片范围进行筛选
参数：
src——源图片
返回值：
图片集
*/
picture siftFeaturematch(Mat src, vector<string> files)
{
	//提取源图片特征
	vector<KeyPoint> src_KP;
	Mat src_des;
	siftFeature128(src, src_des, src_KP);

	//提取训练图片特征
	vector<vector<KeyPoint>> train_KP0 = readFiles_KeyPoint("siftKeyPoint.xml");
	vector<picture> trained0 = readFiles("sift128Vector.xml");

	//选出需要匹配的特征
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