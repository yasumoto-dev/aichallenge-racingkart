#include "Obstacle.h"

using namespace Eigen;
using namespace std;

// Bounded 2D line-segment intersection test, replacing the original
// QLineF::intersect()-based implementation (dropped to remove the Qt5
// dependency; this is a standard orientation-based segment intersection
// test and preserves the same "do these two bounded segments cross"
// semantics as QLineF::IntersectType::BoundedIntersection).
namespace {
double cross2d(double ox, double oy, double ax, double ay, double bx, double by) {
    return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox);
}

bool bounded_segments_intersect(double p1x, double p1y, double p2x, double p2y,
                                 double p3x, double p3y, double p4x, double p4y) {
    double d1 = cross2d(p3x, p3y, p4x, p4y, p1x, p1y);
    double d2 = cross2d(p3x, p3y, p4x, p4y, p2x, p2y);
    double d3 = cross2d(p1x, p1y, p2x, p2y, p3x, p3y);
    double d4 = cross2d(p1x, p1y, p2x, p2y, p4x, p4y);
    return ((d1 > 0) != (d2 > 0)) && (d1 != 0 && d2 != 0) &&
           ((d3 > 0) != (d4 > 0)) && (d3 != 0 && d4 != 0);
}
}  // namespace

Obstacle::Obstacle(Vector2f first_point, Vector2f second_point, double obstacle_clearance)
{
    // Get topLeft and bottomRight points from the given points.
    Vector2f tmp;
    if (first_point.x() > second_point.x() && first_point.y() > second_point.y()) {
        tmp = first_point;
        first_point = second_point;
        second_point = tmp;
    } else if (first_point.x() < second_point.x() && first_point.y() > second_point.y()) {
        float height = first_point.y() - second_point.y();
        first_point.y() -= height;
        second_point.y() += height;
    } else if (first_point.x() > second_point.x() && first_point.y() < second_point.y()) {
        float length = first_point.x() - second_point.x();
        first_point.x() -= length;
        second_point.x() += length;
    }
    first_point.x() -= obstacle_clearance;
    first_point.y() -= obstacle_clearance;
    second_point.x() += obstacle_clearance;
    second_point.y() += obstacle_clearance;

    bbox.first.x() = first_point.x();
    bbox.first.y() = first_point.y();
    bbox.second.x() = second_point.x();
    bbox.second.y() = second_point.y();
}

// Determine whether given line segment intersects an obstacle
// Arguments:
//      p1: point 1 in the line segment
//      p2: point 2 in the line segment
// Returns:
//      whether given line segment intersects an obstacle
bool Obstacle::isSegmentInObstacle(Vector2f &p1, Vector2f &p2)
{
    float length = bbox.second.x() - bbox.first.x();
    float breadth = bbox.second.y() - bbox.first.y();
    double lseg1x1 = bbox.first.x(), lseg1y1 = bbox.first.y();
    double lseg1x2 = bbox.first.x() + length, lseg1y2 = bbox.first.y();
    double lseg2x1 = bbox.first.x(), lseg2y1 = bbox.first.y();
    double lseg2x2 = bbox.first.x(), lseg2y2 = bbox.first.y() + breadth;
    double lseg3x1 = bbox.second.x(), lseg3y1 = bbox.second.y();
    double lseg3x2 = bbox.second.x(), lseg3y2 = bbox.second.y() - breadth;
    double lseg4x1 = bbox.second.x(), lseg4y1 = bbox.second.y();
    double lseg4x2 = bbox.second.x() - length, lseg4y2 = bbox.second.y();

    bool x1 = bounded_segments_intersect(p1.x(), p1.y(), p2.x(), p2.y(),
                                          lseg1x1, lseg1y1, lseg1x2, lseg1y2);
    bool x2 = bounded_segments_intersect(p1.x(), p1.y(), p2.x(), p2.y(),
                                          lseg2x1, lseg2y1, lseg2x2, lseg2y2);
    bool x3 = bounded_segments_intersect(p1.x(), p1.y(), p2.x(), p2.y(),
                                          lseg3x1, lseg3y1, lseg3x2, lseg3y2);
    bool x4 = bounded_segments_intersect(p1.x(), p1.y(), p2.x(), p2.y(),
                                          lseg4x1, lseg4y1, lseg4x2, lseg4y2);

    return x1 || x2 || x3 || x4;
}

bool Obstacle::isPointNearObstacle(Vector2f &p, double radius) {
    double dist_to_ll, dist_to_lr, dist_to_ul, dist_to_ur;
    dist_to_ll = sqrt(pow(bbox.first.x() - p.x(), 2) +
                      pow(bbox.first.y() - p.y(), 2));
    dist_to_lr = sqrt(pow(bbox.second.x() - p.x(), 2) +
                      pow(bbox.first.y() - p.y(), 2));
    dist_to_ul = sqrt(pow(bbox.first.x() - p.x(), 2) +
                      pow(bbox.second.y() - p.y(), 2));
    dist_to_ur = sqrt(pow(bbox.second.x() - p.x(), 2) +
                      pow(bbox.second.y() - p.y(), 2));
    if (dist_to_ll <= radius || dist_to_lr <= radius ||
        dist_to_ul <= radius || dist_to_ur <= radius ) {
        return true;
    }
    return false;
}