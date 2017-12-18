#include"dog.h"
/*
	�����ͼƬ��lbp��GLCM��16ά����
	������
		a����ͼƬa
	����ֵ��
		1*16������
*/
Mat sixteengrainFeature(Mat src)
{
	CV_Assert(src.channels() == 1);
	CV_Assert(src.rows > 0 && src.cols > 0);

	Mat lbp;
	LBP_invariant(src, lbp);
	Mat feature16 = Mat(1, 16, CV_64FC1);
	int k = 0;
	for (int i = 0; i < 4; i++) {//�ֱ����0��45��90��135�ȵĻҶȹ�������
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
	������ͼƬ���г�ʼ��
	������
		Tem����ͼƬ��
	����ֵ��
		��
*/
void grainFeatureInit(vector<picture>& Tem)
{
	for (int i = 0; i < Tem.size(); i++) {
		cvtColor(Tem[i].pic, Tem[i].pic, COLOR_BGR2GRAY);
		blur(Tem[i].pic, Tem[i].pic, Size(3, 3));
	}
}

/*
	������ͼƬ����ѵ������¼���ص�
	������
		Tem����ͼƬ��
	����ֵ��
		��
*/
void grainFeaturetrain(vector<picture> Tem)
{
	grainFeatureInit(Tem);
	vector<picture> fea16(Tem.size());
	for (int i = 0; i < Tem.size(); i++) {
		fea16[i].pic = sixteengrainFeature(Tem[i].pic);
		fea16[i].name = Tem[i].name;
		cout << "����ѵ����" << i << endl;
	}
	writeFiles("grain.xml", fea16);
}

/*
	��ͼƬ��Χ����ɸѡ
	������
		src����ԴͼƬ
	����ֵ��
		ͼƬ��
*/
vector<picture> grainFeaturematch(Mat src, vector<string> files, double th, int times)
{
	vector<picture> trained = readFiles("grain.xml");
	vector<picture> select = getPicturesFromFiles(trained, files);
	Mat tmp = sixteengrainFeature(src);
	vector<picture> match = matchTemplate_grainFeature(tmp, select, th, times);
	return match;
}