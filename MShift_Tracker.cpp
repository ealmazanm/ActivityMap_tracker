#include "MShift_Tracker.h"


MShift_Tracker::MShift_Tracker(void)
{
	numPeople = 0;
}


MShift_Tracker::~MShift_Tracker(void)
{
}

bool isBiggestNeigh(int val, int row, int col, const Mat* img)
{
	int maxRow = img->rows;
	int maxCols = img->cols;
	bool bigger = true;
	int cRow = row -1;
	while (bigger &&  cRow <= row+1)
	{
		int cCol = col -1;
		if (cRow >=0 && cRow < maxRow)
		{
			const ushort* ptr = img->ptr<ushort>(cRow);
			while (bigger && cCol <= col+1)
			{
				if (cCol >= 0 && cCol < maxCols)
				{
					if (cRow != row || cCol != col)
						bigger = (val > ptr[cCol]);
				}
				cCol++;
			}
		}
		cRow++;
	}
	return bigger;
}


Point getCentroid(const Mat* frame, Rect roi, bool flag)
{
	int numPoints = 0;
	Point mean = Point(0,0);
	int endY = roi.y + roi.height;
	int endX = roi.x + roi.width;
	for (int y = roi.y; y < endY; y++)
	{
		const uchar* imgPtr = frame->ptr<const uchar>(y);
		for (int x = roi.x; x < endX; x++)
		{
			if ((flag && imgPtr[x*3+2] < 240) || (!flag && ((imgPtr[x*3]  < 240) || (imgPtr[x*3+1] < 240)  || (imgPtr[x*3+2] < 240) ))) 
			{
				mean.x += x;
				mean.y += y;
				numPoints++;
			}
		}
	}
	mean.x /= numPoints;
	mean.y /= numPoints;

	return mean;
}

void getDistrParam(const Mat* frame, Rect roi, Point* centr, int* sigmaX, int* sigmaY, bool flag)
{
	double sumR, sumC, sumRR, sumCC;
	sumRR = sumCC = sumR = sumC = 0.0;
	int numPoints = 0;
	int endY = roi.y + roi.height;
	int endX = roi.x + roi.width;
	for (int r = roi.y; r < endY; r++)
	{
		const uchar* imgPtr = frame->ptr<const uchar>(r);
		for (int c = roi.x; c < endX; c++)
		{
			if ((flag && imgPtr[c*3+2] < 240) || (!flag && ((imgPtr[c*3]  < 240) || (imgPtr[c*3+1] < 240)  || (imgPtr[c*3+2] < 240) ))) 
			{
				sumC += c;
				sumR += r;
				sumRR += r*r;
				sumCC += c*c;
				numPoints++;
			}
		}
	}

	centr->x = sumC/numPoints;
	centr->y = sumR/numPoints;
	*sigmaX = (int)sqrtf((sumCC/numPoints) - powf(centr->x, 2));
	*sigmaY = (int)sqrtf((sumRR/numPoints) - powf(centr->y, 2));
}

Rect MShift_Tracker::createROI(const Mat* frame, int r, int c, Size binSz)
{
	int initX = max(0, c-binSz.width);
	int initY = max(0, r-binSz.height);
	int width = min(2*binSz.width, frame->cols - initX);
	int height = min(2*binSz.height, frame->rows - initY);
	return Rect(initX, initY, width, height);
}

void MShift_Tracker::createModel(const Mat* frame, const Mat* mask_roi, const Rect* roi_model, MatND& model, bool flag, int debug)
{	
	Mat img_roi = (*frame)(*roi_model);

	if (debug == DEBUG_HIGH)
	{
		destroyWindow("Roi");
		destroyWindow("Mask");
		imshow("Roi", img_roi);
		imshow("Mask", *mask_roi);
		waitKey(0);
	}


	if (flag)
	{
		//HSV parameters
		int hbins = 30, sbins = 32;
		int histSize_HSV[] = {hbins, sbins}; 
		float hranges[] = { 0, 179}; 
		float sranges[] = { 0, 255};
		const float* ranges_HSV[] = {hranges, sranges}; 
		int channels_HSV[] = {0, 1};
		calcHist(&img_roi, 1, channels_HSV, *mask_roi, model, 2, histSize_HSV, ranges_HSV, true, false);
		normalize(model, model, 0, 255, NORM_MINMAX, -1, Mat());
		//calcBackProject(img, 1, channels_HSV, hist, probImage, ranges_HSV);
	}
	else
	{
		//RGB parameters
		int rbins = 30, gbins = 32, bbins = 32; 
		int histSize_RGB[] = {rbins, gbins, bbins}; 
		float rranges[] = { 0, 256}; 
		float granges[] = { 0, 256 };
		float branges[] = { 0, 256 };
		const float* ranges_RGB[] = {rranges, granges, branges}; 
		int channels_RGB[] = {0, 1, 2};	
		calcHist(&img_roi, 1, channels_RGB, *mask_roi, model, 2, histSize_RGB, ranges_RGB, true, false);
		normalize(model, model, 0, 255, NORM_MINMAX, -1, Mat());
		//calcBackProject(img, 1, channels_RGB, hist, probImage, ranges_RGB);
	}
}

Rect MShift_Tracker::getProbabilityImg(Mat* frame, Mshift_person* p, Mat& probImg, const Mat* mask, bool flag, int debug)
{
	int *channels;
	float **ranges;
	if (!flag)
	{
		//RGB values
 		float rranges[] = { 0, 256}; 
		float granges[] = { 0, 256 };
		float branges[] = { 0, 256 };
		ranges = new float*[3];
		ranges[0] = rranges; ranges[1] = granges; ranges[2] = branges;
//		const float* ranges_RGB[] = {rranges, granges, branges}; 
		channels = new int[3];
		channels[0] = 0; channels[1] = 1; channels[2] =2;
//		int channels_RGB[] = {0, 1, 2};	
	}
	else
	{
		//HSV values
		float hranges[] = { 0, 179}; 
		float sranges[] = { 0, 255};
		ranges = new float*[2];
		ranges[0] = hranges; ranges[1] = sranges;
		//const float* ranges_HSV[] = {hranges, sranges}; 
		channels = new int[2];
		channels[0] = 0; channels[1] = 1;
		//int channels_HSV[] = {0, 1};	
	}
	//only from a search roi area
	probImg = Mat::zeros(frame->rows, frame->cols, CV_8UC1);
	Point centroid = p->getCentroid();
	centroid.x += p->getVelocity().x;
	centroid.y += p->getVelocity().y;
	//checking
	if (centroid.x < 0) centroid.x = 1;
	if (centroid.x >= probImg.cols) centroid.x = probImg.cols-2;
	if (centroid.y < 0) centroid.y = 1;
	if (centroid.y >= probImg.rows) centroid.y = probImg.rows-2;

	Rect roi_search = createROI(&probImg, centroid.y, centroid.x, Size(p->getSigmaX()*1.3, p->getSigmaY()*1.3));
	//Rect roi_search = createROI(&probImg, centroid.y, centroid.x, Size(p->getSigmaX()*1, p->getSigmaY()*1));
	//Checks
	assert(roi_search.x >= 0 && roi_search.x < probImg.cols);
	assert(roi_search.y >= 0 && roi_search.y < probImg.rows);
	assert(roi_search.width > 0 && (roi_search.x + roi_search.width) <= probImg.cols);
	assert(roi_search.height > 0 && (roi_search.y + roi_search.height) <= probImg.rows);

	Mat prob_roi = probImg(roi_search);
	Mat frame_roi = (*frame)(roi_search);

	calcBackProject(&frame_roi, 1, channels, *(p->getModel()), prob_roi, (const float**)ranges);
	Mat out;
	Mat mask_roi = (*mask)(roi_search);
	bitwise_and(prob_roi, mask_roi, out);
	out.copyTo(prob_roi);

	if (debug == MShift_Tracker::DEBUG_HIGH)
	{
		//rectangle(probImg, Point(roi_search.x, roi_search.y), Point(roi_search.x+roi_search.width, roi_search.y+roi_search.height),Scalar(255,105,0));
		imshow("Prob", probImg);
		waitKey(0);
	}

	return roi_search;
}

bool isPerson(const Mat* probImg)
{
	int num = 0;
	for (int i = 0; i < probImg->rows; i++)
	{
		const uchar* ptr = probImg->ptr<uchar>(i);
		for (int j = 0; j < probImg->cols; j++)
		{
			if (ptr[j] > 0)
				num++;
		}
	}
	return (num > 50);
}

//Point updateCov(const Mat* mask, Rect* roi, list<Mshift_person>::iterator pers)
Point updateCov(const Mat* mask, list<Mshift_person>::iterator pers)
{
	int sumX, sumY, sumXY, sumXX, sumYY, total;
	total = sumXX = sumYY = sumX = sumY = sumXY = 0;
	float varX, varY, varXY;

//	int endY = roi->y+ roi->height;
//	int endX = roi->x+ roi->width;
	for (int i = 0; i < mask->rows; i++)
	{
		const uchar* ptrMask = mask->ptr<uchar>(i);
		for (int j = 0; j < mask->cols; j++)
		{
			int val = (int)ptrMask[j];
			if (val > 0)
			{
				sumX += j;
				sumY += i;
				sumXX += j*j;
				sumYY += i*i;
				sumXY += i*j;
				total++;
			}
		}
	}
	if (total == 0) return Point(-1,-1);
	Point mean = Point (sumX/total, sumY/total);
	varX = (sumXX/total) - powf(mean.x, 2.0);
	varY = (sumYY/total) - powf(mean.y, 2.0);
	varXY = (sumXY/total) - (mean.x*mean.y);
	Matx22f cov(varX, varXY, varXY, varY);

	(*pers).setCovMat(&cov);
	return mean;
}

//todo: check iterator address
void updateModel(const Mat* prob_roi, Mat& img_roi, list<Mshift_person>::iterator pers, int minProb, float alpha, bool flag)
{
	//pixels with a probability higher than minProb
	Mat mask_roi;
	threshold(*prob_roi, mask_roi, minProb, 255, THRESH_BINARY);
	MatND model;
	if (flag)
	{
		//HSV parameters
		int hbins = 30, sbins = 32;
		int histSize_HSV[] = {hbins, sbins}; 
		float hranges[] = { 0, 179}; 
		float sranges[] = { 0, 255};
		const float* ranges_HSV[] = {hranges, sranges}; 
		int channels_HSV[] = {0, 1};
		calcHist(&img_roi, 1, channels_HSV, mask_roi, model, 2, histSize_HSV, ranges_HSV, true, false);
		normalize(model, model, 0, 255, NORM_MINMAX, -1, Mat());
	}
	else
	{
		//RGB parameters
		int rbins = 30, gbins = 32, bbins = 32; 
		int histSize_RGB[] = {rbins, gbins, bbins}; 
		float rranges[] = { 0, 256}; 
		float granges[] = { 0, 256 };
		float branges[] = { 0, 256 };
		const float* ranges_RGB[] = {rranges, granges, branges}; 
		int channels_RGB[] = {0, 1, 2};	
		calcHist(&img_roi, 1, channels_RGB, mask_roi, model, 2, histSize_RGB, ranges_RGB, true, false);
		normalize(model, model, 0, 255, NORM_MINMAX, -1, Mat());
	}

	(*pers).setModel(&model, alpha);
}

//Reduce the amount of points in the binned data. 
//only counts those points with a prob higher than "minProb"
void cleanData(Mat* prob, Mat& img, Mat& fullMask, Rect* roi, Mat& spatialHist, Size binSz, int minProb, bool flag, int debug)
{	
	if (debug == MShift_Tracker::DEBUG_MED)
	{
		imshow("Mask debug", fullMask);
		waitKey(0);
	}
	for (int i = roi->y; i < roi->y+roi->height; i++)
	{
		uchar* ptrImg = img.ptr<uchar>(i);
		uchar* ptrProb = prob->ptr<uchar>(i);
		uchar* ptrMask = fullMask.ptr<uchar>(i);
		for (int j = roi->x; j < roi->x+roi->width; j++)
		{
			int prob = (int)ptrProb[j];
			int maskVal = (int)ptrMask[j];
			if (maskVal != 0 && prob > minProb)
			{
				//clean image
				if (flag)
				{
					ptrImg[j*3] = 0; ptrImg[j*3+1] = 0; ptrImg[j*3+2] = 255;
				}
				else
				{
					ptrImg[j*3] = 255; ptrImg[j*3+1] = 255; ptrImg[j*3+2] = 255;
				}

				//clean binned image
				int colBin = (int)j/binSz.width;
				int rowBin = (int)i/binSz.height;
				
				if (rowBin < 0 || rowBin >= spatialHist.rows || colBin < 0 || colBin >= spatialHist.cols)
				{
					if (rowBin == spatialHist.rows)
						rowBin--;
					else if (colBin == spatialHist.cols)
						colBin--;
					else
						cout << "Error bin counting" << endl;

				}

				//updates the histogram and mask
				ushort* ptrBin = spatialHist.ptr<ushort>(rowBin);
				int val = (int)ptrBin[colBin];
				if (val == 0)
					cout << "Error cleaning the data: Bincol: " << colBin << ", BinRow: " << rowBin << endl;
				else
					ptrBin[colBin]--;

				ptrMask[j] = 0;
				ptrProb[j] = 0;
			}
		}
	}

	if (debug >= MShift_Tracker::DEBUG_MED)
	{
		Mat frame = Mat::zeros(img.size(), CV_8UC1);
		for (int c = 0; c < spatialHist.cols; c++)
				line(frame, Point(c*binSz.width, 0), Point(c*binSz.width, frame.rows), Scalar::all(255));

		for (int r = 0; r < spatialHist.rows; r++)
				line(frame, Point(0, r*binSz.height), Point(frame.cols, r*binSz.height), Scalar::all(255));
	
		for (int i = 0; i < spatialHist.rows; i++)
		{
			ushort* ptr = spatialHist.ptr<ushort>(i);
			for (int j = 0; j < spatialHist.cols; j++)
			{
				int num = (int)ptr[j];
				int col = j*binSz.width;
				int row = i*binSz.height;
				if (num > 0)
				{
					char txt[15];
					itoa(num, txt, 10);
					putText(frame, txt, Point(col+(binSz.width/2.5),row + (binSz.height/2)),FONT_HERSHEY_PLAIN, 0.8, Scalar::all(255));
				}
				else
					putText(frame, "0", Point(col+(binSz.width/2.5),row + (binSz.height/2)),FONT_HERSHEY_PLAIN, 0.8, Scalar::all(255));
			}
		}
		imshow("hist clean", frame);
		imshow("Mask debugII", fullMask);
		waitKey(0);
		destroyWindow("hist clean");
		destroyWindow("Mask debug");
		destroyWindow("Mask debugII");
	}
}

Point calculateMaxMin(Point p1, Point p2)
{
	Point pOut = Point(-1,-1);
	if ((p1.x <= p2.x) && (p1.y > p2.x))
	{
		pOut.x = p2.x;
		pOut.y = std::min(p1.y, p2.y);
	}
	else if ((p1.x >= p2.x) && (p1.x < p2.y))
	{
		pOut.x = p1.x;
		pOut.y = std::min(p1.y, p2.y);
	}
	return pOut;
}

Rect calculateOverlap(const Rect* roi1, const Rect* roi2)
{
	Rect out = Rect(-1,-1, -1,-1);
	Point px = calculateMaxMin(Point(roi1->x, roi1->x+roi1->width), Point(roi2->x, roi2->x+roi2->width));
	Point py = calculateMaxMin(Point(roi1->y, roi1->y+roi1->height), Point(roi2->y, roi2->y+roi2->height));
	if (px.x != -1 && py.x != -1)
	{
		out.x = px.x;
		out.y = py.x;
		out.width = px.y-px.x;
		out.height = py.y-py.x;
	}
	return out;
}


void MShift_Tracker::updateProbs(const Mat* mask, Mat& prob1, Mat& prob2, const Rect* roi, Mshift_person* p, Mshift_person* p2, int debug)
{

	Mat tmp = Mat::zeros(roi->height, roi->width, CV_8UC3);
	if (debug == MShift_Tracker::DEBUG_HIGH)
	{
		Mat p1_roi = prob1(*roi);
		Mat p2_roi = prob2(*roi);
		Mat p1_clo = Mat::zeros(p1_roi.rows, p1_roi.cols, CV_8UC1);
		Mat p2_clo = Mat::zeros(p1_roi.rows, p1_roi.cols, CV_8UC1);
		Mat mask_roi = (*mask)(*roi);
		for (int i = 0; i < roi->height; i++)
		{
			uchar* p1ClonePtr = p1_clo.ptr<uchar>(i);
			uchar* p2ClonePtr = p2_clo.ptr<uchar>(i);
			uchar* p1Ptr = p1_roi.ptr<uchar>(i);
			uchar* p2Ptr = p2_roi.ptr<uchar>(i);
			uchar* tmpPtr = tmp.ptr<uchar>(i);
			uchar* maskPtr = mask_roi.ptr<uchar>(i);
			for (int j = 0; j < roi->width; j++)
			{
				int val = (int)maskPtr[j];
				if (val > 0)
				{
					tmpPtr[j*3] = 255;
					tmpPtr[j*3 + 1] = 255;
					tmpPtr[j*3 + 2] = 255;
					if (p1Ptr[j] > 0)
						p1ClonePtr[j] = 255;
					if (p2Ptr[j] > 0)
						p2ClonePtr[j] = 255;
				}
			}
		}
		imshow("p1", p1_clo);
		//imshow("p2", p2_clo);
		waitKey(0);
	}

	//Mat prob1_roi = prob1(*roi);
	//Mat prob2_roi = prob2(*roi);
	Point centr1 = p->getCentroid();
	Point centr2 = p2->getCentroid();

	Matx22f covMatInv1 = p->getCovMat()->inv();
	Matx22f covMatInv2 = p2->getCovMat()->inv();

	int endY = roi->y + roi->height;
	int endX = roi->x + roi->width;
	for (int i = roi->y; i < endY; i++)
	//for (int i = 0 ; i < prob1_roi.rows; i++)
	{
		uchar* ptr1 = prob1.ptr<uchar>(i);
		uchar* ptr2 = prob2.ptr<uchar>(i);
		const uchar* maskPtr = mask->ptr<uchar>(i);
		for (int j = roi->x; j < endX; j++)
		{
			int maskVal = (int)maskPtr[j];
			int probVal1 = (int)ptr1[j];
			int probVal2 = (int)ptr2[j];
			if (maskVal > 0)// && probVal1 > 0 && probVal2 > 0)
			{
  				Matx12f diff1(j-centr1.x, i-centr1.y);
				Matx12f diff2(j-centr2.x, i-centr2.y);

				Mat res1 = (Mat)diff1 * (Mat)covMatInv1 * (Mat)diff1.t();
				Mat res2 = (Mat)diff2 * (Mat)covMatInv2 * (Mat)diff2.t();

				float dist1 = sqrtf((float)res1.at<float>(0));
				float dist2 = sqrtf((float)res2.at<float>(0));
				if (dist1 < dist2)
				{
					ptr2[j] = 0;
					if (debug == MShift_Tracker::DEBUG_HIGH)
					{
							int rowTmp = i-roi->y;
							int colTmp = j-roi->x;
							uchar* tmpPtr = tmp.ptr<uchar>(rowTmp);
							tmpPtr[colTmp*3] = 255;
							tmpPtr[colTmp*3+1] = 0;
							tmpPtr[colTmp*3+2] = 0;
							imshow("distances", tmp);
							waitKey(0);
					}
				}
				else
				{
					ptr1[j] = 0;
					if (debug == MShift_Tracker::DEBUG_HIGH)
					{
							int rowTmp = i-roi->y;
							int colTmp = j-roi->x;
							uchar* tmpPtr = tmp.ptr<uchar>(rowTmp);
							tmpPtr[colTmp*3] = 0;
							tmpPtr[colTmp*3+1] = 0;
							tmpPtr[colTmp*3+2] = 255;
							imshow("distances", tmp);
							waitKey(0);
					}
				}
			}
		}
	}
	if (debug == MShift_Tracker::DEBUG_HIGH)
	{
		Mat p1_roi = prob1(*roi);
		Mat p2_roi = prob2(*roi);
		Mat p1_clo = Mat::zeros(p1_roi.rows, p1_roi.cols, CV_8UC1);
		Mat p2_clo = Mat::zeros(p1_roi.rows, p1_roi.cols, CV_8UC1);
		Mat mask_roi = (*mask)(*roi);
		for (int i = 0; i < roi->height; i++)
		{
			uchar* p1ClonePtr = p1_clo.ptr<uchar>(i);
			uchar* p2ClonePtr = p2_clo.ptr<uchar>(i);
			uchar* p1Ptr = p1_roi.ptr<uchar>(i);
			uchar* p2Ptr = p2_roi.ptr<uchar>(i);
			uchar* tmpPtr = tmp.ptr<uchar>(i);
			uchar* maskPtr = mask_roi.ptr<uchar>(i);
			for (int j = 0; j < roi->width; j++)
			{
				int val = (int)maskPtr[j];
				if (val > 0)
				{
					tmpPtr[j*3] = 255;
					tmpPtr[j*3 + 1] = 255;
					tmpPtr[j*3 + 2] = 255;
					if (p1Ptr[j] > 0)
						p1ClonePtr[j] = 255;
					if (p2Ptr[j] > 0)
						p2ClonePtr[j] = 255;
				}
			}
		}
		imshow("p1", p1_clo);
		//imshow("p2", p2_clo);
		waitKey(0);
		destroyWindow("distances");
	}
}
void updateProbs(Mat& prob1, Mat& prob2, const Rect* roi)
{
	Mat prob1_roi = prob1(*roi);
	Mat prob2_roi = prob2(*roi);
	for (int i = 0 ; i < prob1_roi.rows; i++)
	{
		uchar* ptr1 = prob1_roi.ptr<uchar>(i);
		uchar* ptr2 = prob2_roi.ptr<uchar>(i);
		for (int j = 0; j < prob1_roi.cols; j++)
		{
			int val1 = (int)ptr1[j];
			int val2 = (int)ptr2[j];
			if (val1 > val2)
				ptr2[j] = 0;
			else
				ptr1[j] = 0;
		}
	}

}

void MShift_Tracker::meanShift(Mat& orig, Mat& frame, Mat& fullMask, Mat& spatialHist, Size binSz, list<Mshift_person>* people, bool flag, int debug)
{
	int minProb = 0;
	int minProb_lost = -1;
	TermCriteria term = TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 50, 0.1);
	//track previous people
	list<Mshift_person>::iterator peopleIter = people->begin();
	Mat* probImgs = new Mat[people->size()];
	Rect* rois = new Rect[people->size()];
	int iter = 0;
	//create a probability image for every person
	while (peopleIter != people->end())
	{ 
		Mshift_person p = *peopleIter;
		if (!p.isLost())
		{
			Mat probImg;
			rois[iter] = getProbabilityImg(&frame, &p, probImg, &fullMask, flag, debug);
			probImgs[iter] = probImg;
		}
		iter++;
		peopleIter++;
	}

	if (debug == MShift_Tracker::DEBUG_MED)
	{
		Mat probBw;
		threshold(probImgs[0], probBw, 0, 255, CV_THRESH_BINARY);
		rectangle(orig, rois[0], Scalar::all(190));
		imshow("prob", probBw);
		imshow("wind_orig", orig);
		waitKey(0);
		destroyWindow("prob");
	}

	peopleIter = people->begin();
	iter = 0;
	while (peopleIter != people->end())
	{ 
		Rect roi_search = rois[iter];
		Mat probImg = probImgs[iter];
		Mshift_person p = *peopleIter;
		if (!p.isLost())
		{		
			//Update probabilities (in case of overlapping regions)
			//In the overlapped region the pixels are assign to the person with
			//the closest Bhattacharyya distance (taking into account the distribution skew)
			list<Mshift_person>::iterator iterPersTmp = peopleIter;
			iterPersTmp++;
			int iterTmp = iter + 1; //start from the next
			while ((iterTmp < people->size()))
			{
				Mshift_person p2 = *iterPersTmp;
				if (!p2.isLost())
				{
					//obtained overlapping area
					Rect overlapRoi = calculateOverlap(&roi_search, &rois[iterTmp]);
					//update probImgs
					if (overlapRoi.x != -1)
					{			
						if (debug == MShift_Tracker::DEBUG_HIGH)
						{
							rectangle(orig, roi_search ,Scalar(0,255,255));
							rectangle(orig, rois[iterTmp] ,Scalar(0,255,255));
							rectangle(orig, overlapRoi ,Scalar(0,0,255));
							imshow("ActMap", orig);
							waitKey(0);
						}
						updateProbs(&fullMask, probImg, probImgs[iterTmp], &overlapRoi, &p, &p2, debug);
					}
				}
				iterPersTmp++;
				iterTmp++;
			}

			//apply mean shift
 			CamShift(probImg, roi_search, term);
			//check and update new position
			if (roi_search.x > 0)
			{
				//it limits the size of the roi 
				if (roi_search.width > 100)
					roi_search.width = 100;
				if (roi_search.height > 100)
					roi_search.height = 100;

				//update person
				//updated the model and clean data (with "high" probability)
				Mat prob_roi = probImg(roi_search);
				Mat frame_roi = frame(roi_search);
				Mat mask_roi = fullMask(roi_search);
				MatND newModel;
			
				if (debug == MShift_Tracker::DEBUG_MED)
				{
 					Mat test = prob_roi.clone();
					Mat testbw;
					threshold(test, testbw, 0, 255, CV_THRESH_BINARY);
					imshow("roi_prob", testbw);
					waitKey(0);
					destroyWindow("roi_prob");
				}

				//updateModel(&prob_roi, frame_roi, peopleIter, minProb, 0.1, flag);
				updateModel(&mask_roi, frame_roi, peopleIter, minProb, 0.2, flag);
			
				(*peopleIter).setRoi(&roi_search);

				//update covariance (with all for. points)
				//Point newCentroid = updateCov(&fullMask, &roi_search, peopleIter);
				Point newCentroid = updateCov(&mask_roi, peopleIter);
				if (newCentroid.x = -1)
				{
					newCentroid.x = roi_search.x + roi_search.width/2;
					newCentroid.y = roi_search.y + roi_search.height/2;
				}
				else
				{
					newCentroid.x += roi_search.x;
					newCentroid.y += roi_search.y;
				}
				//update direction
				//Point newCentroid = Point(roi_search.x+roi_search.width/2, roi_search.y+roi_search.height/2);
				Point newDirect = Point(newCentroid.x-p.getCentroid().x, newCentroid.y-p.getCentroid().y);
				(*peopleIter).setVelocity(newDirect);
				
				//update framelost
				(*peopleIter).resetFrameLost();

				//update centroid
				(*peopleIter).setCentroid(newCentroid);

				//update sigmaX, sigmaY
				(*peopleIter).setSigmaX(roi_search.width);
				(*peopleIter).setSigmaY(roi_search.height);

				cleanData(&probImg, frame, fullMask, &roi_search, spatialHist, binSz, minProb, flag, debug);

				//peopleIter++;
			}
			else
			{
				cout << "target not found" << endl;
				if (p.getFramesLost() > 5)
				{
					//remove person from list
					//peopleIter = people->erase(peopleIter);
					//set status to lost
					(*peopleIter).setLost();
					cout << "target lost" << endl;
				}
				else
				{
					cout << "target updated with velocity" << endl;
					//update centroid and roi with last velocity
					Point newCentr = Point(p.getCentroid().x+p.getVelocity().x, p.getCentroid().y+p.getVelocity().y);
					if ((newCentr.x >= 0 && newCentr.x < frame.cols) && (newCentr.y >= 0) && (newCentr.y < frame.rows))
					{
						Rect roi_search2 = createROI(&frame, newCentr.y, newCentr.x, Size(p.getSigmaX(), p.getSigmaY()));
						Mat prob_roi = probImg(roi_search2);
						Mat frame_roi = frame(roi_search2);
						//update model and clean
						const Mat mask_roi = fullMask(roi_search2);
						updateModel(&mask_roi, frame_roi, peopleIter, 240 , 0.1, flag);//update with all the for. points in the search area

						cleanData(&probImgs[iter], frame, fullMask, &roi_search2, spatialHist, binSz, minProb_lost, flag, debug);
						(*peopleIter).setCentroid(newCentr);
					}
					(*peopleIter).addFrameLost();
				
				}
			}
		}
		peopleIter++;
		iter++;
	}
	delete []probImgs;
	delete []rois;
	//search for new people
	//Non maxima supression
	Rect roi_search;
	for (int i = 0; i < spatialHist.rows; i++)
	{
		ushort* ptr = spatialHist.ptr<ushort>(i);
		for (int j = 0; j < spatialHist.cols; j++)
		{
			//x,y position of the activityMap (top left corner of the bin)
			int col = j*binSz.width + (binSz.width/2);
			int row = i*binSz.height + (binSz.height/2);

			int num = (int)ptr[j];
			if (num > NON_MAXIMA_THRESHOLD && isBiggestNeigh(num, i, j, &spatialHist))
			{
				//Define the region to search the person
  				roi_search = createROI(&frame, row, col, binSz);
 				Point centroid;
				int sigmaX, sigmaY;
				getDistrParam(&frame, roi_search, &centroid, &sigmaX, &sigmaY, flag);
				//Point centroid = getCentroid(&frame, roi_min, flag);
				Rect roi_model = createROI(&frame, centroid.y, centroid.x, Size(sigmaX*2, sigmaY*2)); // 2std deviations (95% of data)
				MatND model;
				const Mat roi_mask = fullMask(roi_model);
				createModel(&frame, &roi_mask, &roi_model, model, flag, debug);

				//create new person
				Mshift_person p;
				p.setCentroid(centroid);
				p.setSigmaX(sigmaX*2);
				p.setSigmaY(sigmaY*2);
				p.setModel(&model);

				p.setId(numPeople++);
				//if (people->size() == 0)
				//	p.setId(1);
				//else
				//	p.setId(people->back().getId()+1);
				
				people->push_back(p);
			}
		}
	}

}



//void MShift_Tracker::meanShift(Mat& frame, Mat& fullMask, Mat& spatialHist, Size binSz, list<Mshift_person>* people, bool flag, int debug)
//{
//	int minProb = 0;
//	int minProb_lost = -1;
//	Rect roi_search;
//	TermCriteria term = TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 50, 0.1);
//	//track previous people
//	list<Mshift_person>::iterator peopleIter = people->begin();
//	while (peopleIter != people->end())
//	{ 
//		Mshift_person p = *peopleIter;
//		//create probability image
//		Mat probImg;
//		Rect roi_search = getProbabilityImg(&frame, &p, probImg, &fullMask, flag, debug);
//
//		//Todo: Check overlapping
//
//		//apply mean shift
// 		CamShift(probImg, roi_search, term);
//		//check and update new position
//		if (roi_search.x > 0)
//		{
//			//update person
//			//updated the model and clean data (with "high" probability)
//			Mat prob_roi = probImg(roi_search);
//			Mat frame_roi = frame(roi_search);
//			Mat mask_roi = fullMask(roi_search);
//			MatND newModel;
//			//updateModel(&prob_roi, frame_roi, peopleIter, minProb, 0.1, flag);
//			updateModel(&mask_roi, frame_roi, peopleIter, minProb, 0.2, flag);
//			cleanData(&probImg, frame, fullMask, &roi_search, spatialHist, binSz, minProb, flag, debug);
//			
//			//update direction
//			Point newCentroid = Point(roi_search.x+roi_search.width/2, roi_search.y+roi_search.height/2);
//			Point newDirect = Point(newCentroid.x-p.getCentroid().x, newCentroid.y-p.getCentroid().y);
//			(*peopleIter).setVelocity(newDirect);
//				
//			//update framelost
//			(*peopleIter).resetFrameLost();
//
//			//update centroid
//			(*peopleIter).setCentroid(newCentroid);
//
//			//update sigmaX, sigmaY
//			(*peopleIter).setSigmaX(roi_search.width);
//			(*peopleIter).setSigmaY(roi_search.height);
//
//			peopleIter++;
//		}
//		else
//		{
//			cout << "target not found" << endl;
//			if (p.getFramesLost() > 5)
//			{
//				//remove person from list
//				peopleIter = people->erase(peopleIter);
//				cout << "target erased" << endl;
//			}
//			else
//			{
//				cout << "target updated with velocity" << endl;
//				//update centroid and roi with last velocity
//				Point newCentr = Point(p.getCentroid().x+p.getVelocity().x, p.getCentroid().y+p.getVelocity().y);
//				if ((newCentr.x >= 0 && newCentr.x < frame.cols) && (newCentr.y >= 0) && (newCentr.y < frame.rows))
//				{
//					Rect roi_search = createROI(&frame, newCentr.y, newCentr.x, Size(p.getSigmaX(), p.getSigmaY()));
//					Mat prob_roi = probImg(roi_search);
//					Mat frame_roi = frame(roi_search);
//					//update model and clean
//					const Mat mask_roi = fullMask(roi_search);
//					updateModel(&mask_roi, frame_roi, peopleIter, 240 , 0.1, flag);//update with all the for. points in the search area
//
//					cleanData(&probImg, frame, fullMask, &roi_search, spatialHist, binSz, minProb_lost, flag, debug);
//					(*peopleIter).setCentroid(newCentr);
//				}
//				(*peopleIter).addFrameLost();
//				peopleIter++;
//			}
//		}
//	}
//
//	//search for new people
//	//Non maxima supression
//	for (int i = 0; i < spatialHist.rows; i++)
//	{
//		ushort* ptr = spatialHist.ptr<ushort>(i);
//		for (int j = 0; j < spatialHist.cols; j++)
//		{
//			//x,y position of the activityMap (top left corner of the bin)
//			int col = j*binSz.width + (binSz.width/2);
//			int row = i*binSz.height + (binSz.height/2);
//
//			int num = (int)ptr[j];
//			if (num > NON_MAXIMA_THRESHOLD && isBiggestNeigh(num, i, j, &spatialHist))
//			{
//				//Define the region to search the person
//  				roi_search = createROI(&frame, row, col, binSz);
// 				Point centroid;
//				int sigmaX, sigmaY;
//				getDistrParam(&frame, roi_search, &centroid, &sigmaX, &sigmaY, flag);
//				//Point centroid = getCentroid(&frame, roi_min, flag);
//				Rect roi_model = createROI(&frame, centroid.y, centroid.x, Size(sigmaX*2, sigmaY*2)); // 2std deviations (95% of data)
//				MatND model;
//				const Mat roi_mask = fullMask(roi_model);
//				createModel(&frame, &roi_mask, &roi_model, model, flag, debug);
//
//				//create new person
//				Mshift_person p;
//				p.setCentroid(centroid);
//				p.setSigmaX(sigmaX*2);
//				p.setSigmaY(sigmaY*2);
//				p.setModel(&model);
//
//				if (people->size() == 0)
//					p.setId(1);
//				else
//					p.setId(people->back().getId()+1);
//				
//				people->push_back(p);
//			}
//		}
//	}
//
//}