#include<opencv2\core\core.hpp>
#include<opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>
#include<iostream>
using namespace std;
using namespace cv;

/*数据结构区*/
typedef struct picture
{
	string name;
	Mat pic;
}picture;
