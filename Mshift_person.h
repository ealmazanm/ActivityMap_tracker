#pragma once
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <ctype.h>

using namespace std;
using namespace cv;


class Mshift_person
{
public:
	Mshift_person(void);
	Mshift_person(int id, MatND* model);
	~Mshift_person(void);

	//Getters
	inline int getId(){ return id; }
	inline Point getCentroid(){ return centroid; }
	inline const MatND* getModel(){ return &model; }
	inline int getFramesLost(){ return framesLost; }
	inline Point getVelocity(){ return velocity; }
	inline int getSigmaX(){ return sigmaX; }
	inline int getSigmaY(){ return sigmaY; }
	inline Matx22f* getCovMat(){ return &cov; }
	inline bool isLost(){return lost;}

	inline Rect getRoi(){return roi;}

	//Setters
	inline void setId(int id_){ id = id_;}
	inline void setCentroid(Point centr_){centroid = centr_;}
	inline void setModel(MatND* model_, float alpha){model = (1-alpha)*model + alpha*(*model_);}
	inline void setModel(MatND* model_){model = (*model_);}
	inline void addFrameLost(){framesLost++;}
	inline void resetFrameLost(){framesLost = 0;}
	inline void setVelocity(Point vel_){velocity = vel_;}
	inline void setSigmaX(int sX){sigmaX = sX;}
	inline void setSigmaY(int sY){sigmaY = sY;}
	inline void setCovMat(Matx22f* c){ cov = *c;}
	inline void setLost(){lost = true;}
	inline void setRoi(Rect *roi_){ roi = *roi_;}

private:
	int id;
	Point centroid;
	MatND model;
	int framesLost;
	Point velocity;
	int sigmaX, sigmaY;
	Matx22f cov; //covariance matrix
	bool lost;

	//test
	Rect roi;
};

