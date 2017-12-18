#include<opencv2\core\core.hpp>
#include<opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>
#include<math.h>
#include<map>
#include<iostream>
#define s(a,b) a>=b?1:0
using namespace std;
using namespace cv;
/*
	利用LBP方法提取图片纹理特征
	采用旋转不变的uniform LBP   共P+1个编码值
	其中，p = 8，R = 1
	参数：
		src——源图片
		det——纹理特征图片
*/
void LBP(Mat src, Mat& det)
{
	CV_Assert(src.channels() == 1);
	CV_Assert(src.depth() != sizeof(uchar));
	CV_Assert(src.rows > 3 && src.cols > 3);

	det = Mat(src.rows, src.cols, CV_8UC1);

	int x, y;
	uchar *pAbove;
	uchar *pCurr;
	uchar *pBelow;
	uchar *nw, *no, *ne;    // north (pAbove)
	uchar *we, *me, *ea;
	uchar *sw, *so, *se;    // south (pBelow)

	uchar *pDst;

	// initialize row pointers
	pAbove = NULL;
	pCurr = src.ptr<uchar>(0);
	pBelow = src.ptr<uchar>(1);

	for (y = 1; y < src.rows - 1; ++y) {
		// shift the rows up by one
		pAbove = pCurr;
		pCurr = pBelow;
		pBelow = src.ptr<uchar>(y + 1);

		pDst = det.ptr<uchar>(y);

		// initialize col pointers
		no = &(pAbove[0]);
		ne = &(pAbove[1]);
		me = &(pCurr[0]);
		ea = &(pCurr[1]);
		so = &(pBelow[0]);
		se = &(pBelow[1]);

		for (x = 1; x < src.cols - 1; ++x) {
			// shift col pointers left by one (scan left to right)
			nw = no;
			no = ne;
			ne = &(pAbove[x + 1]);
			we = me;
			me = ea;
			ea = &(pCurr[x + 1]);
			sw = so;
			so = se;
			se = &(pBelow[x + 1]);

			int U_lbp = 
				abs(s(*we, *me) - s(*nw, *me)) + abs(s(*no, *me) - s(*nw, *me)) +
				abs(s(*ne, *me) - s(*no, *me)) + abs(s(*ea, *me) - s(*ne, *me)) +
				abs(s(*se, *me) - s(*ea, *me)) + abs(s(*so, *me) - s(*se, *me)) +
				abs(s(*sw, *me) - s(*so, *me)) + abs(s(*we, *me) - s(*sw, *me));
			int lbp = U_lbp > 2 ?
				9 :
				s(*nw, *me) + s(*no, *me) +
				s(*ne, *me) + s(*ea, *me) +
				s(*se, *me) + s(*so, *me) +
				s(*sw, *me) + s(*we, *me);

			lbp *= 10;
			pDst[x] = (uchar)lbp;
		}
	}
}

/*
	利用LBP方法提取图片纹理特征
	采用旋转不变的 LBP   共36个编码值
	其中，p = 8，R = 1
	参数：
		src——源图片
		det——纹理特征图片
*/
map<uchar, uchar> initmap();
void LBP_invariant(Mat src, Mat& det)
{
	CV_Assert(src.channels() == 1);
	CV_Assert(src.depth() != sizeof(uchar));
	CV_Assert(src.rows > 3 && src.cols > 3);

	det = Mat(src.rows, src.cols, CV_8UC1);
	map<uchar,uchar> m = initmap();

	int x, y;
	uchar *pAbove;
	uchar *pCurr;
	uchar *pBelow;
	uchar *nw, *no, *ne;    // north (pAbove)
	uchar *we, *me, *ea;
	uchar *sw, *so, *se;    // south (pBelow)

	uchar *pDst;

	// initialize row pointers
	pAbove = NULL;
	pCurr = src.ptr<uchar>(0);
	pBelow = src.ptr<uchar>(1);

	for (y = 1; y < src.rows - 1; ++y) {
		// shift the rows up by one
		pAbove = pCurr;
		pCurr = pBelow;
		pBelow = src.ptr<uchar>(y + 1);

		pDst = det.ptr<uchar>(y);

		// initialize col pointers
		no = &(pAbove[0]);
		ne = &(pAbove[1]);
		me = &(pCurr[0]);
		ea = &(pCurr[1]);
		so = &(pBelow[0]);
		se = &(pBelow[1]);

		for (x = 1; x < src.cols - 1; ++x) {
			// shift col pointers left by one (scan left to right)
			nw = no;
			no = ne;
			ne = &(pAbove[x + 1]);
			we = me;
			me = ea;
			ea = &(pCurr[x + 1]);
			sw = so;
			so = se;
			se = &(pBelow[x + 1]);

			int lbp =
				(s(*we, *me)<<7) + (s(*no, *me)<<6) +
				(s(*ne, *me)<<5) + (s(*ea, *me)<<4) +
				(s(*se, *me)<<3) + (s(*so, *me)<<2) +
				(s(*sw, *me)<<1) + (s(*we, *me)<<0);

			pDst[x] = m[(uchar)lbp];
		}
	}
}

map<uchar, uchar> initmap()
{
	map<uchar, uchar> m;
	for (uchar c = 0; c < 256; c++) {
		uchar min = c;
		for (int i = 1; i <= 8; i++) {//右移
			uchar tmp1 = c, tmp2 = c;
			uchar tmp = uchar(tmp1 >> (8 - i)) + uchar(tmp2 << i);
			if (tmp < min) min = tmp;
		}
		m[c] = min;
		if (c == 255)break;
	}
	return m;
}