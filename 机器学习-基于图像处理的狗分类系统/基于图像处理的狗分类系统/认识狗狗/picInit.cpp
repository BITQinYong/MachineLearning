// Grabcut.cpp : �������̨Ӧ�ó������ڵ㡣  
//  

//#include "stdafx.h"  




#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/imgproc/imgproc.hpp"  

#include <iostream>  
#include "ComputeTime.h"  
#include "windows.h"  

using namespace std;
using namespace cv;
Mat photo;
static void help()
{
	cout << "\nThis program demonstrates GrabCut segmentation -- select an object in a region\n"
		"and then grabcut will attempt to segment it out.\n"
		"Call:\n"
		"./grabcut <image_name>\n"
		"\nSelect a rectangular area around the object you want to segment\n" <<
		"\nHot keys: \n"
		"\tESC - quit the program\n"
		"\tr - restore the original image\n"
		"\tn - next iteration\n"
		"\n"
		"\tleft mouse button - set rectangle\n"
		"\n"
		"\tCTRL+left mouse button - set GC_BGD pixels\n"
		"\tSHIFT+left mouse button - set CG_FGD pixels\n"
		"\n"
		"\tCTRL+right mouse button - set GC_PR_BGD pixels\n"
		"\tSHIFT+right mouse button - set CG_PR_FGD pixels\n" << endl;
}

const Scalar RED = Scalar(0, 0, 255);
const Scalar PINK = Scalar(230, 130, 255);
const Scalar BLUE = Scalar(255, 0, 0);
const Scalar LIGHTBLUE = Scalar(255, 255, 160);
const Scalar GREEN = Scalar(0, 255, 0);

const int BGD_KEY = CV_EVENT_FLAG_CTRLKEY;  //Ctrl��  
const int FGD_KEY = CV_EVENT_FLAG_SHIFTKEY; //Shift��  
Rect rect1, rect2;
static void getBinMask(const Mat& comMask, Mat& binMask)
{
	if (comMask.empty() || comMask.type() != CV_8UC1)
		CV_Error(CV_StsBadArg, "comMask is empty or has incorrect type (not CV_8UC1)");
	if (binMask.empty() || binMask.rows != comMask.rows || binMask.cols != comMask.cols)
		binMask.create(comMask.size(), CV_8UC1);
	binMask = comMask & 1;  //�õ�mask�����λ,ʵ������ֻ����ȷ���Ļ����п��ܵ�ǰ���㵱��mask  
}

class GCApplication
{
public:
	enum { NOT_SET = 0, IN_PROCESS = 1, SET = 2 };
	static const int radius = 2;
	static const int thickness = -1;

	void reset();
	void setImageAndWinName(const Mat& _image, const string& _winName);
	void showImage() const;
	void mouseClick(int event, int x, int y, int flags, void* param);
	int nextIter();
	int getIterCount() const { return iterCount; }
private:
	void setRectInMask();
	void setLblsInMask(int flags, Point p, bool isPr);

	const string* winName;
	const Mat* image;
	Mat mask;
	Mat bgdModel, fgdModel;

	uchar rectState, lblsState, prLblsState;
	bool isInitialized;

	Rect rect;
	vector<Point> fgdPxls, bgdPxls, prFgdPxls, prBgdPxls;
	int iterCount;
};

/*����ı�����ֵ*/
void GCApplication::reset()
{
	if (!mask.empty())
		mask.setTo(Scalar::all(GC_BGD));
	bgdPxls.clear(); fgdPxls.clear();
	prBgdPxls.clear();  prFgdPxls.clear();

	isInitialized = false;
	rectState = NOT_SET;    //NOT_SET == 0  
	lblsState = NOT_SET;
	prLblsState = NOT_SET;
	iterCount = 0;
}

/*����ĳ�Ա������ֵ����*/
void GCApplication::setImageAndWinName(const Mat& _image, const string& _winName)
{
	if (_image.empty() || _winName.empty())
		return;
	image = &_image;
	winName = &_winName;
	mask.create(image->size(), CV_8UC1);
	reset();
}

/*��ʾ4���㣬һ�����κ�ͼ�����ݣ���Ϊ����Ĳ���ܶ�ط���Ҫ�õ�������������Ե����ó���*/
void GCApplication::showImage() const
{
	if (image->empty() || winName->empty())
		return;

	Mat res;
	Mat binMask;
	//binMask.create(image->cols,image->rows, CV_8U,Scalar(0,255,0));
	if (!isInitialized)
		image->copyTo(res);
	else
	{
		getBinMask(mask, binMask);
		image->copyTo(res, binMask);  //�������λ��0����1�����ƣ�ֻ������ǰ���йص�ͼ�񣬱���˵���ܵ�ǰ�������ܵı���  
	}

	vector<Point>::const_iterator it;
	/*����4������ǽ�ѡ�е�4�����ò�ͬ����ɫ��ʾ����*/
	for (it = bgdPxls.begin(); it != bgdPxls.end(); ++it)  //���������Կ�����һ��ָ��  
		circle(res, *it, radius, BLUE, thickness);
	for (it = fgdPxls.begin(); it != fgdPxls.end(); ++it)  //ȷ����ǰ���ú�ɫ��ʾ  
		circle(res, *it, radius, RED, thickness);
	for (it = prBgdPxls.begin(); it != prBgdPxls.end(); ++it)
		circle(res, *it, radius, LIGHTBLUE, thickness);
	for (it = prFgdPxls.begin(); it != prFgdPxls.end(); ++it)
		circle(res, *it, radius, PINK, thickness);

	/*������*/
	if (rectState == IN_PROCESS || rectState == SET)
		rectangle(res, Point(rect.x, rect.y), Point(rect.x + rect.width, rect.y + rect.height), GREEN, 2);
	for (int i = 0; i < res.rows; i++) {
		uchar *date = res.ptr(i);
		for (int j = 0; j < res.cols; j++) {

			if (date[j * 3 + 0] == 0 && date[j * 3 + 1] == 0 && date[j * 3 + 2] == 0) {
				date[j * 3 + 0] = 180;
				date[j * 3 + 1] = 227;
				date[j * 3 + 2] = 27;
				//date[j * 3 + 0] = 73;
				//date[j * 3 + 1] = 190;
				//date[j * 3 + 2] = 114;
			}
		}
	}
	photo = res;
	imshow(*winName, res);


}

/*�ò�����ɺ�maskͼ����rect�ڲ���3������ȫ��0*/
void GCApplication::setRectInMask()
{
	assert(!mask.empty());
	mask.setTo(GC_BGD);   //GC_BGD == 0  
	rect.x = max(0, rect.x);
	rect.y = max(0, rect.y);
	rect.width = min(rect.width, image->cols - rect.x);
	rect.height = min(rect.height, image->rows - rect.y);
	(mask(rect)).setTo(Scalar(GC_PR_FGD));    //GC_PR_FGD == 3�������ڲ�,Ϊ���ܵ�ǰ����  
}

void GCApplication::setLblsInMask(int flags, Point p, bool isPr)
{
	vector<Point> *bpxls, *fpxls;
	uchar bvalue, fvalue;
	if (!isPr) //ȷ���ĵ�  
	{
		bpxls = &bgdPxls;
		fpxls = &fgdPxls;
		bvalue = GC_BGD;    //0  
		fvalue = GC_FGD;    //1  
	}
	else    //���ʵ�  
	{
		bpxls = &prBgdPxls;
		fpxls = &prFgdPxls;
		bvalue = GC_PR_BGD; //2  
		fvalue = GC_PR_FGD; //3  
	}
	if (flags & BGD_KEY)
	{
		bpxls->push_back(p);
		circle(mask, p, radius, bvalue, thickness);   //�õ㴦Ϊ2  
	}
	if (flags & FGD_KEY)
	{
		fpxls->push_back(p);
		circle(mask, p, radius, fvalue, thickness);   //�õ㴦Ϊ3  
	}
}

/*�����Ӧ����������flagsΪCV_EVENT_FLAG�����*/
void GCApplication::mouseClick(int event, int x, int y, int flags, void*)
{
	// TODO add bad args check  
	switch (event)
	{
	case CV_EVENT_LBUTTONDOWN: // set rect or GC_BGD(GC_FGD) labels  
	{
		bool isb = (flags & BGD_KEY) != 0,
			isf = (flags & FGD_KEY) != 0;
		if (rectState == NOT_SET && !isb && !isf)//ֻ���������ʱ  
		{
			rectState = IN_PROCESS; //��ʾ���ڻ�����  
			rect = Rect(x, y, 1, 1);
			//rect1 = rect;
			//cout <<"1--"<< rect << endl;
		}
		if ((isb || isf) && rectState == SET) //������alt������shift�����һ����˾��Σ���ʾ���ڻ�ǰ��������  
			lblsState = IN_PROCESS;
	}
	break;
	case CV_EVENT_RBUTTONDOWN: // set GC_PR_BGD(GC_PR_FGD) labels  
	{
		bool isb = (flags & BGD_KEY) != 0,
			isf = (flags & FGD_KEY) != 0;
		if ((isb || isf) && rectState == SET) //���ڻ����ܵ�ǰ��������  
			prLblsState = IN_PROCESS;
	}
	break;
	case CV_EVENT_LBUTTONUP:
		if (rectState == IN_PROCESS)
		{
			rect = Rect(Point(rect.x, rect.y), Point(x, y));   //���ν���  
			rect2 = rect;
			//cout <<"2--"<< rect << endl;
			rectState = SET;
			setRectInMask();
			assert(bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty());
			showImage();
		}
		if (lblsState == IN_PROCESS)   //�ѻ���ǰ�󾰵�  
		{
			setLblsInMask(flags, Point(x, y), false);    //����ǰ����  
			lblsState = SET;
			showImage();
		}
		break;
	case CV_EVENT_RBUTTONUP:
		if (prLblsState == IN_PROCESS)
		{
			setLblsInMask(flags, Point(x, y), true); //����������  
			prLblsState = SET;
			showImage();
		}
		break;
	case CV_EVENT_MOUSEMOVE:
		if (rectState == IN_PROCESS)
		{
			rect = Rect(Point(rect.x, rect.y), Point(x, y));
			//cout <<"3--"<< rect << endl;
			assert(bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty());
			showImage();    //���ϵ���ʾͼƬ  
		}
		else if (lblsState == IN_PROCESS)
		{
			setLblsInMask(flags, Point(x, y), false);
			showImage();
		}
		else if (prLblsState == IN_PROCESS)
		{
			setLblsInMask(flags, Point(x, y), true);
			showImage();
		}
		break;
	}
}

/*�ú�������grabcut�㷨�����ҷ����㷨���е����Ĵ���*/
int GCApplication::nextIter()
{
	if (isInitialized)
		//ʹ��grab�㷨����һ�ε���������2Ϊmask��������maskλ�ǣ������ڲ�������Щ�����Ǳ��������Ѿ�ȷ���Ǳ���������еĵ㣬��maskͬʱҲΪ���  
		//������Ƿָ���ǰ��ͼ��  
		grabCut(*image, mask, rect, bgdModel, fgdModel, 1);
	else
	{
		if (rectState != SET)
			return iterCount;

		if (lblsState == SET || prLblsState == SET)
			grabCut(*image, mask, rect, bgdModel, fgdModel, 1, GC_INIT_WITH_MASK);
		else
			grabCut(*image, mask, rect, bgdModel, fgdModel, 1, GC_INIT_WITH_RECT);

		isInitialized = true;
	}
	iterCount++;



	bgdPxls.clear(); fgdPxls.clear();
	prBgdPxls.clear(); prFgdPxls.clear();

	return iterCount;
}

GCApplication gcapp;

static void on_mouse(int event, int x, int y, int flags, void* param)
{
	gcapp.mouseClick(event, x, y, flags, param);
}

Mat koutu(Mat image)
{


	Mat ph; 

	//help();

	const string winName = "image";
	cvNamedWindow(winName.c_str(), CV_WINDOW_AUTOSIZE);
	cvSetMouseCallback(winName.c_str(), on_mouse, 0);

	gcapp.setImageAndWinName(image, winName);
	gcapp.showImage();
	cvWaitKey(0);
	for (;;)
	{
		int c = cvWaitKey(0);
		cout << " you have inputed the char '" << (char)c << "'" << endl;
		switch ((char)c)
		{
		case 'e':
			rect2.x+=2; rect2.y+=2;
			rect2.width -= 3; rect2.height -= 3;
			ph = photo(rect2);
			return ph;
		case '\x1b':
			cout << "Exiting ..." << endl;
			goto exit_main;
		case 'r':
			cout << endl;
			gcapp.reset();
			gcapp.showImage();
			break;
		case 'n':
			ComputeTime ct;
			ct.Begin();

			int iterCount = gcapp.getIterCount();
			cout << "<" << iterCount << "... ";
			int newIterCount = gcapp.nextIter();
			if (newIterCount > iterCount)
			{
				gcapp.showImage();
				cout << iterCount << ">" << endl;
				cout << "����ʱ��:  " << ct.End() << endl;
			}
			else
				cout << "rect must be determined>" << endl;
			break;
		}
	}

exit_main:
	cvDestroyWindow(winName.c_str());
	return photo;
}