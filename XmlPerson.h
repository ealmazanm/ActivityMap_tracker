#pragma once
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <ctype.h>
#include <list>

using namespace std;
using namespace cv;

class XmlPerson
{
public:
	XmlPerson(void);
	~XmlPerson(void);
	
	inline int getInitFrame(){ return initFrame; }
	inline int getEndFrame(){ return endFrame; }
	inline list<Matx22d>* getLocations(){ return &locations;}
	inline int getId(){return id; }

	inline void setInitFrame(int n){ initFrame = n;}
	inline void setEndFrame(int n){ endFrame = n;}
	inline void addLocation(Matx22d* loc){locations.push_back(*loc);}
	inline void setId(int id_){ id = id_;}


private:
	int initFrame;
	int endFrame;
	list<Matx22d> locations;//(1,:) init and end frames. (2,:) position
	int id;
};

