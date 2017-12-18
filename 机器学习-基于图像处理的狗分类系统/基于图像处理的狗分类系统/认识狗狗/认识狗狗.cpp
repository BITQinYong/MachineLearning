#include"dog.h"
bool haveTrained = true;
int main()
{
	/*初始化*/
	vector<string> files;
	getFiles(".\\dogss", "", files);//获取文件名
	vector<picture> pics;
	getPictures(files, pics);//获取图片

	/*进入程序*/
	while (1) {
		/*纹理特征训练*/
		if (haveTrained == false) {
			grainFeaturetrain(pics);
			colorFeaturetrain(pics);
			shapeFeaturetrain(pics);
			siftFeaturetrain(pics);
			haveTrained = true;
		}

		/*输入图片及预处理*/
		string filename;
		cin >> filename;
		Mat src = imread(filename), src1;
		if (src.data == NULL) {
			return 0;
		}

		src = koutu(src);
		imshow("src", src);
		cvWaitKey(0);

		cout << "颜色" << endl;
		/*颜色特征匹配*/
		vector<picture> colorMatchPics = colorFeaturematch(src, files, 0.7, 3);
		vector<string> colorFiles = getFilesFromPictures(files, colorMatchPics);

		for (int i = 0; i < colorFiles.size(); i++) {
			cout << colorFiles[i] << endl;
		}


		cout << "形状"<<endl;
		/*形状特征匹配*/
		vector<picture> shapeMatchPics = shapeFeaturematch(src, colorFiles, 0.3, 1);
		vector<string> shapeFiles = getFilesFromPictures(colorFiles, shapeMatchPics);
		for (int i = 0; i < shapeFiles.size(); i++) {
			cout << shapeFiles[i] << endl;
		}


		cvtColor(src, src1, CV_BGR2GRAY);
		blur(src1, src1, Size(3, 3));

		cout << "纹理" << endl;
		/*纹理特征匹配*/
		vector<picture> grainMatchPics = grainFeaturematch(src1, shapeFiles, 0.7, 1);
		vector<string> grainFiles = getFilesFromPictures(shapeFiles, grainMatchPics);

		for (int i = 0; i < grainFiles.size(); i++) {
			cout << grainFiles[i] << endl;
		}
		cout << endl;

		/*sift特征匹配*/
		picture ans = siftFeaturematch(src, grainFiles);

		/*显示结果*/
		imshow(ans.name, ans.pic);
		waitKey(0);
	}
}