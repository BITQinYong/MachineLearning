#pragma once  
/*
//������������ʱ�����
//
*/
#include <iostream>  

#ifndef ComputeTime_h  
#define ComputeTime_h  


class   ComputeTime
{
private:
	int Initialized;
	__int64 Frequency;
	__int64 BeginTime;

public:

	bool Avaliable();
	double End();
	bool Begin();
	ComputeTime();
	virtual   ~ComputeTime();

};






#endif  


#include "ComputeTime.h"  
#include <iostream>  
#include <Windows.h>  

ComputeTime::ComputeTime()
{
	Initialized = QueryPerformanceFrequency((LARGE_INTEGER   *)&Frequency);
}

ComputeTime::~ComputeTime()
{

}

bool   ComputeTime::Begin()
{
	if (!Initialized)
		return 0;

	return   QueryPerformanceCounter((LARGE_INTEGER   *)&BeginTime);
}

double   ComputeTime::End()
{
	if (!Initialized)
		return 0;


	__int64   endtime;

	QueryPerformanceCounter((LARGE_INTEGER   *)&endtime);


	__int64   elapsed = endtime - BeginTime;


	return   ((double)elapsed / (double)Frequency)*1000.0;  //��λ����  
}

bool   ComputeTime::Avaliable()
{
	return Initialized;
}
