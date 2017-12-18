#include"dog.h"
/*
	计算出图片经lbp和GLCM后16维特征
	参数：
		a——图片a
	返回值：
		1*16的特征
*/
Mat sixteengrainFeature(Mat src)
{
	CV_Assert(src.channels() == 1);
	CV_Assert(src.rows > 0 && src.cols > 0);

	Mat lbp;
	LBP_invariant(src, lbp);
	Mat feature16 = Mat(1, 16, CV_64FC1);
	int k = 0;
	for (int i = 0; i < 4; i++) {//分别计算0、45、90、135度的灰度共生矩阵
		double angle = i * 45;
		Mat glcm = GLCM(lbp, 1.5, angle);
		vector<double> tmp = GLCM_Haralick(glcm);
		for (int j = 0; j < 4; j++) {
			feature16.at<Vec<double, 1>>(0, k++)[0] = tmp[j];
		}
	}

	return feature16;
}

/*
	对已有图片进行初始化
	参数：
		Tem——图片集
	返回值：
		无
*/
void grainFeatureInit(vector<picture>& Tem)
{
	for (int i = 0; i < Tem.size(); i++) {
		cvtColor(Tem[i].pic, Tem[i].pic, COLOR_BGR2GRAY);
		blur(Tem[i].pic, Tem[i].pic, Size(3, 3));
	}
}

/*
	对已有图片进行训练，记录其特点
	参数：
		Tem——图片集
	返回值：
		无
*/
void grainFeaturetrain(vector<picture> Tem)
{
	grainFeatureInit(Tem);
	vector<picture> fea16(Tem.size());
	for (int i = 0; i < Tem.size(); i++) {
		fea16[i].pic = sixteengrainFeature(Tem[i].pic);
		fea16[i].name = Tem[i].name;
		cout << "纹理训练：" << i << endl;
	}
	writeFiles("grain.xml", fea16);
}

/*
	对图片范围进行筛选
	参数：
		src——源图片
	返回值：
		图片集
*/
vector<picture> grainFeaturematch(Mat src, vector<string> files, double th, int times)
{
	vector<picture> trained = readFiles("grain.xml");
	vector<picture> select = getPicturesFromFiles(trained, files);
	Mat tmp = sixteengrainFeature(src);
	vector<picture> match = matchTemplate_grainFeature(tmp, select, th, times);
	return match;
}