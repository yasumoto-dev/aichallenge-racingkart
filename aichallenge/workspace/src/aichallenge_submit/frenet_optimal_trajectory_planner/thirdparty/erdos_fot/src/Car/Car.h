#ifndef FRENETOPTIMALTRAJECTORY_CAR_H
#define FRENETOPTIMALTRAJECTORY_CAR_H

#include "utils.h"

#include <vector>
#include <tuple>

using namespace std;

// Local patch (racing_kart_description/config/vehicle_info.param.yaml):
// wheel_base 1.087 + front_overhang 0.467 + rear_overhang 0.510 = 2.064 length,
// wheel_tread 1.12 + left/right_overhang 0.09*2 = 1.30 width. Upstream hardcodes
// a Lincoln MKZ (4.93 x 1.86) here for the hard collision-check footprint; that
// is not exposed via FrenetHyperparameters, so it must be edited directly for
// any other vehicle.
const double VEHICLE_LENGTH = 2.064;
const double VEHICLE_WIDTH = 1.30;

class Car {
public:
    Car() {
        length = VEHICLE_LENGTH;
        width = VEHICLE_WIDTH;
    };
    Car(Pose pose_): pose(pose_) {
        length = VEHICLE_LENGTH;
        width = VEHICLE_WIDTH;
    };
    void setPose(Pose p);
    vector<Point> getOutline();
private:
    double length;
    double width;
    Pose pose; // x, y, yaw
};

#endif //FRENETOPTIMALTRAJECTORY_CAR_H
