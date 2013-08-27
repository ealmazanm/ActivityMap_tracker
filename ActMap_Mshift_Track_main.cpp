#include "Mshift_person.h"
#include <ActivityMap_Utils.h>
#include "MShift_Tracker.h"
#include <list>
#include "XmlPerson.h"
#include <tinystr.h>
#include <tinyxml.h>

using namespace std;
using namespace cv;

const static int NUM_COLS = 20;
const static int NUM_ROWS = 10;


int debug = MShift_Tracker::DEBUG_LOW;
const bool HSV_DATA = true;

int main()
{
	//load video
	//VideoCapture inSeq("D:\\Emilio\\Tracking\\DataSet\\s_1pers\\MoA_1pers.avi");
	//VideoCapture inSeq("D:\\Emilio\\Tracking\\DataSet\\s_2pers_nOccl\\MoA_2pers_nOccl.avi");
	//VideoCapture inSeq("D:\\Emilio\\Tracking\\DataSet\\s_2pers_occl\\MoA_2pers_occl.mpg");
	VideoCapture inSeq("D:\\Emilio\\Tracking\\DataSet\\Dset2_workshop\\MoA_dst2workshop.mpg");
		
	
//	ofstream out1("D:\\Debug\\debug1.txt", ios::out);
//	ofstream out2("D:\\Debug\\debug2.txt", ios::out);
	
	list<Mshift_person> people;
	list<XmlPerson> xmlPeople;
	
	//loop through all frames
	Mat frame, hsv;
	Size binSize;
	bool first = true;

	VideoWriter	w;

	MShift_Tracker mshiftTracker;

	int frames  = 0;
	while (inSeq.grab())
	{
		cout << "Frames: " << frames << endl;
//		if (frames == 191)
//			DEBUG = MShift_Tracker::DEBUG_MED;
//			cout << "debug" << endl;

		inSeq.retrieve(frame);

		

		cvtColor(frame, hsv, CV_BGR2HSV );
		Mat gray;
		cvtColor(frame, gray, CV_BGR2GRAY);
		Mat full_mask;
		threshold(gray, full_mask, 230, 255, CV_THRESH_BINARY_INV);
		if (first)
		{
			binSize.width = hsv.cols/NUM_COLS;
			binSize.height = hsv.rows/NUM_ROWS;

			w.open("D:\\Emilio\\Tracking\\DataSet\\Dset2_workshop\\MoA_dst2workshop_Tracked.mpg",CV_FOURCC('P','I','M','1'), 20.0, frame.size() , true);
			first = false;
		}
		//Data bining
		Mat frame_binned = Mat::zeros(NUM_ROWS, NUM_COLS, CV_16U); //ushort
		ActivityMap_Utils::getImageBinned(&full_mask, frame_binned, binSize, HSV_DATA);
		
		if (debug > MShift_Tracker::DEBUG_NONE)
		{
			//draw the bins
			if (debug == MShift_Tracker::DEBUG_MED)
			{
				for (int c = 0; c < NUM_COLS; c++)
					line(frame, Point(c*binSize.width, 0), Point(c*binSize.width, frame.rows), Scalar::all(0));

				for (int r = 0; r < NUM_ROWS; r++)
					line(frame, Point(0, r*binSize.height), Point(frame.cols, r*binSize.height), Scalar::all(0));
			}
			//draw the value of each bin
			if (debug == MShift_Tracker::DEBUG_MED)
			{
				for (int i = 0; i < NUM_ROWS; i++)
				{
					ushort* ptr = frame_binned.ptr<ushort>(i);
					for (int j = 0; j < NUM_COLS; j++)
					{
						int num = (int)ptr[j];
						int col = j*binSize.width;
						int row = i*binSize.height;
						if (num > 0)
						{
							char txt[15];
							itoa(num, txt, 10);
							putText(frame, txt, Point(col+(binSize.width/2.5),row + (binSize.height/2)),FONT_HERSHEY_PLAIN, 0.8, Scalar(30, 0, 255));
						}
						else
							putText(frame, "0", Point(col+(binSize.width/2.5),row + (binSize.height/2)),FONT_HERSHEY_PLAIN, 0.8, Scalar(30, 0, 255));
					}
				}
			}
		}

		//MeanShift tracking
		//MShift_Tracker::meanShift(frame, hsv, full_mask, frame_binned, binSize, &people, HSV_DATA, debug);
		mshiftTracker.meanShift(frame, hsv, full_mask, frame_binned, binSize, &people, HSV_DATA, debug);
		if (debug > MShift_Tracker::DEBUG_NONE)
		{
			//show the full mask;
			if (debug == MShift_Tracker::DEBUG_HIGH)
				cv::imshow("Full mask", full_mask);

			if (debug >= MShift_Tracker::DEBUG_LOW)
			{
				int cont = 0;
				list<Mshift_person>::iterator iter = people.begin();
				while (iter != people.end())
				{
					Mshift_person p = *iter;
					if (!p.isLost())
					{
						cont++;
						circle(frame, p.getCentroid(), 2, Scalar(255,0,0), 2);
						//Rect roi_model = MShift_Tracker::createROI(&frame, p.getCentroid().y, p.getCentroid().x, Size(p.getSigmaX(), p.getSigmaY())); 
						//rectangle(frame, Point(roi_model.x, roi_model.y), Point(roi_model.x+roi_model.width, roi_model.y+roi_model.height),Scalar(255,105,0));
						rectangle(frame, p.getRoi(),Scalar(255,105,0));
						char txt[15];
						itoa(p.getId(), txt, 10);
						//putText(frame, txt, Point(roi_model.x+10,roi_model.y+10),FONT_HERSHEY_PLAIN, 0.7, Scalar(255,105,0));
						putText(frame, txt, Point(p.getRoi().x+5,p.getRoi().y+10),FONT_HERSHEY_PLAIN, 0.7, Scalar(0,0,255));
					}
					iter++;
				}
				cout << "Number of people found: " << cont << endl;
			}
		}

		int nPeop = people.size();
		if (nPeop > 0)
		{
			if (nPeop > xmlPeople.size())
			{
				for (int i = xmlPeople.size(); i < nPeop; i++)
				{
					XmlPerson xmlp;
					xmlPeople.push_back(xmlp);
				}
			}
			list<XmlPerson>::iterator iterXml = xmlPeople.begin();
			list<Mshift_person>::iterator iter = people.begin();
			while (iter != people.end())
			{
				Mshift_person p = *iter;
				if (!p.isLost())
				{
					Point centroid = p.getCentroid();
					if ((*iterXml).getInitFrame() == -1)
						(*iterXml).setInitFrame(frames);
					(*iterXml).setEndFrame(frames);
				
					list<Matx22d>* locs = (*iterXml).getLocations();
					if (locs->size() > 0)//compare current and last position
					{
						Matx22d* prev = &(locs->back());
						if (((*prev)(1,0) == centroid.x) && ((*prev)(1,1) == centroid.y))//udpate last location
						{
							(*prev)(0,1) = frames;
						}
						else //create a new location
						{
							Matx22d newLoc(frames, frames, centroid.x, centroid.y);
							(*iterXml).addLocation(&newLoc);
						}
					}
					else //create a new location
					{
						Matx22d newLoc(frames, frames, centroid.x, centroid.y);
						(*iterXml).addLocation(&newLoc);
					}

					(*iterXml).setId(p.getId());
				}
				iterXml++;
				iter++;
			}
		}

		w << frame;
		imshow("ActMap", frame);
		if (people.empty())
			waitKey(1);
		else if (debug == MShift_Tracker::DEBUG_MED)
			waitKey(0);
		else
			waitKey(100);
		

		frames++;
		//Draw people
	}

	//Create xml file
	TiXmlDocument doc("D:\\Emilio\\Tracking\\DataSet\\Dset2_workshop\\test1.xml");
	bool loadOkay = doc.LoadFile();
	if (!loadOkay) exit;

	list<XmlPerson>::iterator iterXml = xmlPeople.begin();
	while (iterXml != xmlPeople.end())
	{
		XmlPerson p = *iterXml;

/*		<object framespan="1:1 72:84" id="1" name="Person">
                <attribute name="Centroid">
                    <data:point framespan="72:72" x="559" y="471" />
                </attribute>
*/
		TiXmlElement* root = doc.FirstChildElement();
		cout << "root name: " << root->Value() << endl;
		TiXmlElement* data = root->FirstChildElement("data");
		TiXmlElement* sfile = data->FirstChildElement("sourcefile");

		TiXmlElement * element = new TiXmlElement( "object" );
		sfile->LinkEndChild(element);

		char txtStr[200];
		//strcpy(txtStr, "1:1 ");
		char tmp[10];
		itoa(p.getInitFrame(), tmp, 10);
		strcpy(txtStr, tmp);
		strcat(txtStr, ":");
		itoa(p.getEndFrame(), tmp, 10);
		strcat(txtStr, tmp);

		//element->SetAttribute("framespan", "1:1 72:84");
		element->SetAttribute("framespan", txtStr);

		itoa(p.getId(), tmp, 10);
		element->SetAttribute("id", tmp);
		element->SetAttribute("name", "Person");

		TiXmlElement * att = new TiXmlElement( "attribute" );
		element->LinkEndChild(att);
		att->SetAttribute("name", "Position");
		//<data:point framespan="72:72" x="559" y="473"/>

		list<Matx22d>* locs = p.getLocations();
		list<Matx22d>::iterator iterLocs = locs->begin();
		while (iterLocs != locs->end())
		{
			TiXmlElement * elem = new TiXmlElement( "data:point" );
			att->LinkEndChild(elem);

			char frameSpan[50];
			char tt[10];
			itoa((*iterLocs)(0,0),tt, 10);
			strcpy(frameSpan, tt);
			strcat(frameSpan, ":");
			itoa((*iterLocs)(0,1),tt, 10);
			strcat(frameSpan, tt);
			elem->SetAttribute("framespan", frameSpan);

			//Centroid
			itoa((*iterLocs)(1,0), tt, 10);
			elem->SetAttribute("x", tt);
			itoa((*iterLocs)(1,1), tt, 10);
			elem->SetAttribute("y", tt);

		
			iterLocs++;
		}

		iterXml++;
	}
	doc.SaveFile("D:\\Emilio\\Tracking\\DataSet\\Dset2_workshop\\test2.xml");
}