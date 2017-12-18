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
利用LBP方法提取图片纹理特征
采用旋转不变的uniform LBP   共P+1个编码值
其中，p = 8，R = 1
参数：
src——源图片
det——纹理特征图片
作者：
杨隆兴
*/
void LBP(Mat src, Mat& det);

/*
利用LBP方法提取图片纹理特征
采用旋转不变的 LBP   共36个编码值
其中，p = 8，R = 1
参数：
src——源图片
det——纹理特征图片
作者：
杨隆兴
*/
void LBP_invariant(Mat src, Mat& det);




/*
基于Hu不变矩的整体图像匹配
调用了opencv的现有函数
参数：
src——源图片
Tem[]——模板图
numTem——模板数量
返回值：
picture
作者：
杨隆兴
*/
picture matchTemplate_HuMatrix(Mat src, picture Tem[], int numTem);

/*
基于sift和RANSAC进行匹配，根据匹配点的数目进行选择
调用了opencv的现有函数
参数：
src——源图片
Tem——模板图
返回值：
picture
作者：
杨隆兴
*/
picture matchTemplate_SiftRANSAC(Mat src, vector<picture> Tem);

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
作者：
杨隆兴
*/
picture matchTemplate_SiftRANSAC_SAVE(Mat src, Mat src_des, vector<KeyPoint> src_KP, vector<picture> Tem, vector<vector<KeyPoint>> Tem_KP);

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
picture matchTemplate_SiftLowe_SAVE(Mat src, Mat src_des, vector<KeyPoint> src_KP, vector<picture> Tem, vector<vector<KeyPoint>> Tem_KP);

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
vector<picture> matchTemplate_grainFeature(Mat src, vector<picture> Tem, double th, int times);

/*
计算两幅图的马氏距离
调用了opencv的现有函数
参数：
a——图片a
b——图片b
返回值：
马氏距离
作者：
杨隆兴
*/
double Mahalanobis(Mat a, Mat b);

/*
利用灰度共生矩阵方法提取图片纹理特征
参数：
src——源图像
dis——极坐标距离
angle——极坐标角度(不是弧度)
作者：
杨隆兴
*/
Mat GLCM(Mat src, double dis, double angle);

/*
计算灰度共生矩阵的Haralick四个特征
参数：
src——源图片
作者：
杨隆兴
*/
vector<double> GLCM_Haralick(Mat src);

/*
计算两幅图的马氏距离
参数：
a——图片a
b——图片b
返回值：
马氏距离
作者：
杨隆兴
*/
double Mahalanobis(Mat a, Mat b);

/*
获取文件夹下所有文件名,
使用了别人的代码
输入：
path	:	文件夹路径
exd		:   所要获取的文件名后缀，如jpg、png等；如果希望获取所有
文件名, exd = ""
输出：
files	:	获取的文件名列表
*/
void getFiles(string path, string exd, vector<string>& files);

/*
获取图片结构
输入：
files	:	文件夹路径
pic		:   图片结构
*/
void getPictures(vector<string> files, vector<picture>& pics);

/*
根据图片得到其对应的文件名
输入：
files	:	文件夹路径
pics	:   图片
输出：
对应的文件名
*/
vector<string> getFilesFromPictures(vector<string> files, vector<picture> pics);

/*
根据文件路径得到其对应的图片
输入：
pics	:	图片（大范围）
files	:   路径
输出：
对应的图片（小范围）
*/
vector<picture> getPicturesFromFiles(vector<picture> pics, vector<string> files);

/*
将图片信息写入磁盘,以xml方式存储
输入：
path	:	文件夹路径
pics	:   特征图片
作者：
杨隆兴
*/
void writeFiles(string filename, vector<picture> pics);

/*
将特征点信息写入磁盘,以xml方式存储
输入：
path	:	文件夹路径
KP	    :   特征点
作者：
杨隆兴
*/
void writeFiles_KeyPoint(string filename, vector<vector<KeyPoint>> KP);

/*
读出特征图片
输入：
path	:	文件夹路径
返回：
pics	:   特征图片
作者：
杨隆兴
*/
vector<picture> readFiles(string filename);

/*
读出特征点
输入：
path	:	文件夹路径
返回：
特征点
作者：
杨隆兴
*/
vector<vector<KeyPoint>> readFiles_KeyPoint(string filename);

/*
对已有图片进行训练，记录其特点
参数：
Tem——图片集
返回值：
无
作者：
杨隆兴
*/
void grainFeaturetrain(vector<picture> Tem);

/*
对图片范围进行筛选
参数：
src——源图片
th——阈值[0,1]
返回值：
图片集
作者：
杨隆兴
*/
vector<picture> grainFeaturematch(Mat src, vector<string> files, double th, int times);

/*
对已有图片进行训练，记录其sift特点
参数：
Tem——图片集
返回值：
无
作者：
杨隆兴
*/
void siftFeaturetrain(vector<picture> Tem);

/*
对图片范围进行筛选
参数：
src——源图片
返回值：
图片集
作者：
杨隆兴
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

