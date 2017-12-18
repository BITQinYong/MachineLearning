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
	return stream.str();   //�˴�Ҳ������ stream>>string_temp  
}

/*  
	��ȡ�ļ����������ļ���,
	�޸��˱��˵Ĵ���
	���룺
		path	:	�ļ���·��
		exd		:   ��Ҫ��ȡ���ļ�����׺����jpg��png�ȣ����ϣ����ȡ����
					�ļ���, exd = ""
	�����
		pics	:	��ȡ��ͼƬ�б�
*/
void getFiles(string path, string exd, vector<string>& files)
{
	//�ļ����
	long   hFile = 0;
	//�ļ���Ϣ
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
			//������ļ����������ļ���,����֮
			//�������,�����б�
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
	��ȡͼƬ�ṹ
	���룺
		files	:	�ļ���·��
		pic		:   ͼƬ�ṹ
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
	����ͼƬ�õ����Ӧ���ļ���
	���룺
		files	:	�ļ���·������Χ��
		pics	:   ͼƬ
	�����
		��Ӧ���ļ�����С��Χ��
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
	�����ļ�·���õ����Ӧ��ͼƬ
	���룺
		pics	:	ͼƬ����Χ��
		files	:   ·��
	�����
		��Ӧ��ͼƬ��С��Χ��
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
	��ͼƬ��Ϣд�����,��xml��ʽ�洢
	���룺
		path	:	�ļ���·��
		pics	:   ����ͼƬ
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
	����������Ϣд�����,��xml��ʽ�洢
	���룺
		path	:	�ļ���·��
		KP	    :   ������
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
	��������ͼƬ
	���룺
		path	:	�ļ���·��
	���أ�
		pics	:   ����ͼƬ
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
	����������
	���룺
		path	:	�ļ���·��
	���أ�
		������
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
