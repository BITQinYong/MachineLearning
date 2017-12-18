#include <string>
#include <io.h>
#include <vector>
#include <iostream>
#include <fstream>
#include"dog.h"
using namespace std;

string int2str(int int_temp)
{
	stringstream stream;
	stream << int_temp;
	return stream.str();   //此处也可以用 stream>>string_temp  
}

/*  
	获取文件夹下所有文件名,
	修改了别人的代码
	输入：
		path	:	文件夹路径
		exd		:   所要获取的文件名后缀，如jpg、png等；如果希望获取所有
					文件名, exd = ""
	输出：
		pics	:	获取的图片列表
*/
void getFiles(string path, string exd, vector<string>& files)
{
	//文件句柄
	long   hFile = 0;
	//文件信息
	struct _finddata_t fileinfo;
	string pathName, exdName;

	if (0 != strcmp(exd.c_str(), ""))
	{
		exdName = "\\*." + exd;
	}
	else
	{
		exdName = "\\*";
	}

	if ((hFile = _findfirst(pathName.assign(path).append(exdName).c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是文件夹中仍有文件夹,迭代之
			//如果不是,加入列表
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(pathName.assign(path).append("\\").append(fileinfo.name), exd, files);
			}
			else
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					files.push_back(pathName.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

/*  
	获取图片结构
	输入：
		files	:	文件夹路径
		pic		:   图片结构
*/
void getPictures(vector<string> files, vector<picture>& pics)
{
	for (int i = 0; i < files.size(); i++) {
		Mat M = imread(files[i]);
		int start = 0, end = 0;
		for (int j = files[i].length() - 1; j >= 0; j--) {
			if (files[i][j] == '.') {
				end = j - 1;
			}
			if (files[i][j] == '\\') {
				start = j + 1;
				break;
			}
		}
		string name = files[i].substr(start, end - start + 1);
		picture p;
		p.name = name; p.pic = M;
		pics.push_back(p);
	}
}

/*  
	根据图片得到其对应的文件名
	输入：
		files	:	文件夹路径（大范围）
		pics	:   图片
	输出：
		对应的文件名（小范围）
*/
vector<string> getFilesFromPictures(vector<string> files, vector<picture> pics)
{
	vector<string> fs;
	for (int i = 0; i < pics.size(); i++) {
		for (int j = 0; j < files.size(); j++) {
			if (files[j].find(pics[i].name) != string::npos) {
				fs.push_back(files[j]);
				break;
			}
		}
	}
	return fs;
}

/*  
	根据文件路径得到其对应的图片
	输入：
		pics	:	图片（大范围）
		files	:   路径
	输出：
		对应的图片（小范围）
*/
vector<picture> getPicturesFromFiles(vector<picture> pics, vector<string> files)
{
	vector<picture> ps;
	for (int i = 0; i < files.size(); i++) {
		for (int j = 0; j < pics.size(); j++) {
			if (files[i].find(pics[j].name) != string::npos) {
				ps.push_back(pics[j]);
				break;
			}
		}
	}
	return ps;
}

/*
	将图片信息写入磁盘,以xml方式存储
	输入：
		path	:	文件夹路径
		pics	:   特征图片
*/
void writeFiles(string filename, vector<picture> pics)
{
	FileStorage fs;
	fs.open(filename, FileStorage::WRITE);
	fs << "pictures" << "[";
	for (int i = 0; i < pics.size(); i++) {
		fs << pics[i].pic;
	}
	fs << "]";
	fs << "names" << "[";
	for (int i = 0; i < pics.size(); i++) {
		fs << pics[i].name;
	}
	fs << "]";
	fs.release();
}

/*
	将特征点信息写入磁盘,以xml方式存储
	输入：
		path	:	文件夹路径
		KP	    :   特征点
*/
void writeFiles_KeyPoint(string filename, vector<vector<KeyPoint>> KP)
{
	FileStorage fs;
	fs.open(filename, FileStorage::WRITE);
	fs << "num" << (int)KP.size();
	for (int i = 0; i < KP.size(); i++) {
		string name = "KeyPoint" + int2str(i);
		fs << name << "[";
		for (int j = 0; j < KP[i].size(); j++) {
			fs << KP[i][j].angle;
			fs << KP[i][j].class_id;
			fs << KP[i][j].octave;
			fs << KP[i][j].pt.x;
			fs << KP[i][j].pt.y;
			fs << KP[i][j].response;
			fs << KP[i][j].size;
		}
		fs << "]";
	}
	fs.release();
}

/*
	读出特征图片
	输入：
		path	:	文件夹路径
	返回：
		pics	:   特征图片
*/
vector<picture> readFiles(string filename)
{
	FileStorage fs;
	fs.open(filename, FileStorage::READ);
	vector<picture> pics;
	FileNode fn = fs["pictures"];
	FileNode fn1 = fs["names"];
	FileNodeIterator it = fn.begin(), it_end = fn.end();
	FileNodeIterator it1 = fn1.begin(), it_end1 = fn1.end();
	for (; it != it_end&&it1 != it_end1; ++it, ++it1) {
		Mat M;
		(*it) >> M;
		string name;
		(*it1) >> name;
		picture p;
		p.name = name; p.pic = M;
		pics.push_back(p);
	}
	return pics;
}

/*
	读出特征点
	输入：
		path	:	文件夹路径
	返回：
		特征点
*/
vector<vector<KeyPoint>> readFiles_KeyPoint(string filename)
{
	FileStorage fs;
	fs.open(filename, FileStorage::READ);
	vector<vector<KeyPoint>> KP;
	FileNode fn = fs["num"];
	int num = 0;
	fn >> num;
	for (int i = 0; i<num; ++i) {
		vector<KeyPoint> kp;
		string name = "KeyPoint" + int2str(i);
		FileNode fn = fs[name];
		FileNodeIterator it = fn.begin(), it_end = fn.end();
		for (; it != it_end; ++it) {
			KeyPoint p;
			(*(it)) >> p.angle;
			(*(it)) >> p.class_id;
			(*(it)) >> p.octave;
			(*(it)) >> p.pt.x;
			(*(it)) >> p.pt.y;
			(*(it)) >> p.response;
			(*(it)) >> p.size;
			kp.push_back(p);
		}
		KP.push_back(kp);
	}
	return KP;
}
