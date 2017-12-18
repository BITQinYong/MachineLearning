#include<opencv2\core\core.hpp>
#include<opencv2\highgui\highgui.hpp>
#include <opencv2\xfeatures2d.hpp>
#include <opencv2\opencv.hpp>
#include<iostream>
#include <string>
#include <io.h>
#include <vector>
#include"picStruct.h"
using namespace std;
using namespace cv;

/*
����LBP������ȡͼƬ��������
������ת�����uniform LBP   ��P+1������ֵ
���У�p = 8��R = 1
������
src����ԴͼƬ
det������������ͼƬ
���ߣ�
��¡��
*/
void LBP(Mat src, Mat& det);

/*
����LBP������ȡͼƬ��������
������ת����� LBP   ��36������ֵ
���У�p = 8��R = 1
������
src����ԴͼƬ
det������������ͼƬ
���ߣ�
��¡��
*/
void LBP_invariant(Mat src, Mat& det);




/*
����Hu����ص�����ͼ��ƥ��
������opencv�����к���
������
src����ԴͼƬ
Tem[]����ģ��ͼ
numTem����ģ������
����ֵ��
picture
���ߣ�
��¡��
*/
picture matchTemplate_HuMatrix(Mat src, picture Tem[], int numTem);

/*
����sift��RANSAC����ƥ�䣬����ƥ������Ŀ����ѡ��
������opencv�����к���
������
src����ԴͼƬ
Tem����ģ��ͼ
����ֵ��
picture
���ߣ�
��¡��
*/
picture matchTemplate_SiftRANSAC(Mat src, vector<picture> Tem);

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
���ߣ�
��¡��
*/
picture matchTemplate_SiftRANSAC_SAVE(Mat src, Mat src_des, vector<KeyPoint> src_KP, vector<picture> Tem, vector<vector<KeyPoint>> Tem_KP);

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
picture matchTemplate_SiftLowe_SAVE(Mat src, Mat src_des, vector<KeyPoint> src_KP, vector<picture> Tem, vector<vector<KeyPoint>> Tem_KP);

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
vector<picture> matchTemplate_grainFeature(Mat src, vector<picture> Tem, double th, int times);

/*
��������ͼ�����Ͼ���
������opencv�����к���
������
a����ͼƬa
b����ͼƬb
����ֵ��
���Ͼ���
���ߣ�
��¡��
*/
double Mahalanobis(Mat a, Mat b);

/*
���ûҶȹ������󷽷���ȡͼƬ��������
������
src����Դͼ��
dis�������������
angle����������Ƕ�(���ǻ���)
���ߣ�
��¡��
*/
Mat GLCM(Mat src, double dis, double angle);

/*
����Ҷȹ��������Haralick�ĸ�����
������
src����ԴͼƬ
���ߣ�
��¡��
*/
vector<double> GLCM_Haralick(Mat src);

/*
��������ͼ�����Ͼ���
������
a����ͼƬa
b����ͼƬb
����ֵ��
���Ͼ���
���ߣ�
��¡��
*/
double Mahalanobis(Mat a, Mat b);

/*
��ȡ�ļ����������ļ���,
ʹ���˱��˵Ĵ���
���룺
path	:	�ļ���·��
exd		:   ��Ҫ��ȡ���ļ�����׺����jpg��png�ȣ����ϣ����ȡ����
�ļ���, exd = ""
�����
files	:	��ȡ���ļ����б�
*/
void getFiles(string path, string exd, vector<string>& files);

/*
��ȡͼƬ�ṹ
���룺
files	:	�ļ���·��
pic		:   ͼƬ�ṹ
*/
void getPictures(vector<string> files, vector<picture>& pics);

/*
����ͼƬ�õ����Ӧ���ļ���
���룺
files	:	�ļ���·��
pics	:   ͼƬ
�����
��Ӧ���ļ���
*/
vector<string> getFilesFromPictures(vector<string> files, vector<picture> pics);

/*
�����ļ�·���õ����Ӧ��ͼƬ
���룺
pics	:	ͼƬ����Χ��
files	:   ·��
�����
��Ӧ��ͼƬ��С��Χ��
*/
vector<picture> getPicturesFromFiles(vector<picture> pics, vector<string> files);

/*
��ͼƬ��Ϣд�����,��xml��ʽ�洢
���룺
path	:	�ļ���·��
pics	:   ����ͼƬ
���ߣ�
��¡��
*/
void writeFiles(string filename, vector<picture> pics);

/*
����������Ϣд�����,��xml��ʽ�洢
���룺
path	:	�ļ���·��
KP	    :   ������
���ߣ�
��¡��
*/
void writeFiles_KeyPoint(string filename, vector<vector<KeyPoint>> KP);

/*
��������ͼƬ
���룺
path	:	�ļ���·��
���أ�
pics	:   ����ͼƬ
���ߣ�
��¡��
*/
vector<picture> readFiles(string filename);

/*
����������
���룺
path	:	�ļ���·��
���أ�
������
���ߣ�
��¡��
*/
vector<vector<KeyPoint>> readFiles_KeyPoint(string filename);

/*
������ͼƬ����ѵ������¼���ص�
������
Tem����ͼƬ��
����ֵ��
��
���ߣ�
��¡��
*/
void grainFeaturetrain(vector<picture> Tem);

/*
��ͼƬ��Χ����ɸѡ
������
src����ԴͼƬ
th������ֵ[0,1]
����ֵ��
ͼƬ��
���ߣ�
��¡��
*/
vector<picture> grainFeaturematch(Mat src, vector<string> files, double th, int times);

/*
������ͼƬ����ѵ������¼��sift�ص�
������
Tem����ͼƬ��
����ֵ��
��
���ߣ�
��¡��
*/
void siftFeaturetrain(vector<picture> Tem);

/*
��ͼƬ��Χ����ɸѡ
������
src����ԴͼƬ
����ֵ��
ͼƬ��
���ߣ�
��¡��
*/
picture siftFeaturematch(Mat src, vector<string> files);




vector<picture> matchTemplate_colorFeature(Mat src, vector<picture> Tem, double th, int times);

void colorFeaturetrain(vector<picture> Tem);

double colordis(Mat a, Mat b);
vector<picture> colorFeaturematch(Mat src, vector<string> files, double th, int times);

double shapedis(Mat a, Mat b);
vector<picture> shapeFeaturematch(Mat src, vector<string> files, double th, int times);
vector<picture> matchTemplate_shapeFeature(Mat src, vector<picture> Tem, double th, int times);
void shapeFeaturetrain(vector<picture> Tem);



Mat koutu(Mat image);

