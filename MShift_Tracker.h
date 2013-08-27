#pragma once
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"
#include <list>
#include <iostream>
#include "Mshift_person.h"

using namespace std;
using namespace cv;

const int NON_MAXIMA_THRESHOLD = 350;

class MShift_Tracker
{
public:
	MShift_Tracker(void);
	~MShift_Tracker(void);
	void meanShift(Mat& orig, Mat& frame, Mat& fullMask, Mat& spatialHist, Size binSize, list<Mshift_person>* people, bool flag, int debug);
	static Rect createROI(const Mat* frame, int r, int c, Size binSz);


	//debug levels
	static const int DEBUG_NONE = -1;
	static const int DEBUG_LOW = 0;
	static const int DEBUG_MED = 1;
	static const int DEBUG_HIGH = 2;

private:
	static void updateProbs(const Mat* mask, Mat& prob1, Mat& prob2, const Rect* roi, Mshift_person* p, Mshift_person* p2, int debug);
	static Rect getProbabilityImg(Mat* frame, Mshift_person* p, Mat& probImg, const Mat* mask, bool flag, int debug);
	static void createModel(const Mat* frame, const Mat* mask_roi, const Rect* roi_model, MatND& model, bool flag, int debug);
	int numPeople;
};

