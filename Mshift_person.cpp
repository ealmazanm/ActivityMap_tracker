#include "Mshift_person.h"


Mshift_person::Mshift_person(void)
{
	framesLost = 0;
	velocity = Point(0,0);
	lost = false;
}

Mshift_person::Mshift_person(int id_, MatND* model_)
{
	id = id;
	model = *model_;
	framesLost = 0;
	velocity = Point(0,0);
	lost = false;
}


Mshift_person::~Mshift_person(void)
{
}
