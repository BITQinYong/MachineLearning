#include"dog.h"
bool haveTrained = true;
int main()
{
	/*��ʼ��*/
	vector<string> files;
	getFiles(".\\dogss", "", files);//��ȡ�ļ���
	vector<picture> pics;
	getPictures(files, pics);//��ȡͼƬ

	/*�������*/
	while (1) {
		/*��������ѵ��*/
		if (haveTrained == false) {
			grainFeaturetrain(pics);
			colorFeaturetrain(pics);
			shapeFeaturetrain(pics);
			siftFeaturetrain(pics);
			haveTrained = true;
		}

		/*����ͼƬ��Ԥ����*/
		string filename;
		cin >> filename;
		Mat src = imread(filename), src1;
		if (src.data == NULL) {
			return 0;
		}

		src = koutu(src);
		imshow("src", src);
		cvWaitKey(0);

		cout << "��ɫ" << endl;
		/*��ɫ����ƥ��*/
		vector<picture> colorMatchPics = colorFeaturematch(src, files, 0.7, 3);
		vector<string> colorFiles = getFilesFromPictures(files, colorMatchPics);

		for (int i = 0; i < colorFiles.size(); i++) {
			cout << colorFiles[i] << endl;
		}


		cout << "��״"<<endl;
		/*��״����ƥ��*/
		vector<picture> shapeMatchPics = shapeFeaturematch(src, colorFiles, 0.3, 1);
		vector<string> shapeFiles = getFilesFromPictures(colorFiles, shapeMatchPics);
		for (int i = 0; i < shapeFiles.size(); i++) {
			cout << shapeFiles[i] << endl;
		}


		cvtColor(src, src1, CV_BGR2GRAY);
		blur(src1, src1, Size(3, 3));

		cout << "����" << endl;
		/*��������ƥ��*/
		vector<picture> grainMatchPics = grainFeaturematch(src1, shapeFiles, 0.7, 1);
		vector<string> grainFiles = getFilesFromPictures(shapeFiles, grainMatchPics);

		for (int i = 0; i < grainFiles.size(); i++) {
			cout << grainFiles[i] << endl;
		}
		cout << endl;

		/*sift����ƥ��*/
		picture ans = siftFeaturematch(src, grainFiles);

		/*��ʾ���*/
		imshow(ans.name, ans.pic);
		waitKey(0);
	}
}