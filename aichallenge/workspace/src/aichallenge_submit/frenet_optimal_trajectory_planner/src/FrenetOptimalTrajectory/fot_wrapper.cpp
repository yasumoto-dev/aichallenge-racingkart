#include "FrenetOptimalTrajectory.h"
#include "FrenetPath.h"
#include "py_cpp_struct.h"
#include "CubicSpline2D.h"
#include "utils.h"

#include <stddef.h>
#include <vector>

using namespace std;

// C++ wrapper to expose the FrenetOptimalTrajectory class to python
extern "C" {
    // Compute the frenet optimal trajectory given initial conditions
    // in frenet space.
    //
    // Arguments:
    //      fot_ic (FrenetInitialConditions *):
    //          struct ptr containing relevant initial conditions to compute
    //          Frenet Optimal Trajectory
    //      fot_hp (FrenetHyperparameters *):
    //          struct ptr containing relevant hyperparameters to compute
    //          Frenet Optimal Trajectory
    //      x_path, y_path, speeds (double *):
    //          ptr to storage arrays for Frenet Optimal Trajectory
    //      params (double *):
    //          ptr to store initial conditions for debugging
    //
    // Returns:
    //      1 if successful, 0 if failure
    //      Also stores the Frenet Optimal Trajectory into x_path, y_path,
    //      speeds if it exists
    void run_fot(
            FrenetInitialConditions *fot_ic, FrenetHyperparameters *fot_hp,
            FrenetReturnValues *fot_rv
            ) {
        FrenetOptimalTrajectory fot = FrenetOptimalTrajectory(fot_ic, fot_hp);
        FrenetPath* best_frenet_path = fot.getBestPath();
        if (best_frenet_path && !best_frenet_path->x.empty()){
            fot_rv->success = 1;
            fot_rv->path_length = std::min(best_frenet_path->x.size(), MAX_PATH_LENGTH);
            for (size_t i = 0; i < fot_rv->path_length; i++) {
                fot_rv->x_path[i] = best_frenet_path->x[i];
                fot_rv->y_path[i] = best_frenet_path->y[i];
                fot_rv->speeds[i] = best_frenet_path->s_d[i];
                fot_rv->ix[i] = best_frenet_path->ix[i];
                fot_rv->iy[i] = best_frenet_path->iy[i];
                fot_rv->iyaw[i] = best_frenet_path->iyaw[i];
                fot_rv->d[i] = best_frenet_path->d[i];
                fot_rv->s[i] = best_frenet_path->s[i];
                fot_rv->speeds_x[i] = cos(best_frenet_path->yaw[i]) *
                    fot_rv->speeds[i];
                fot_rv->speeds_y[i] = sin(best_frenet_path->yaw[i]) *
                    fot_rv->speeds[i];
            }


            // store info for debug
            fot_rv->params[0] = best_frenet_path->s[1];
            fot_rv->params[1] = best_frenet_path->s_d[1];
            fot_rv->params[2] = best_frenet_path->d[1];
            fot_rv->params[3] = best_frenet_path->d_d[1];
            fot_rv->params[4] = best_frenet_path->d_dd[1];

            // store costs for logging
            fot_rv->costs[0] = best_frenet_path->c_lateral_deviation;
            fot_rv->costs[1] = best_frenet_path->c_lateral_velocity;
            fot_rv->costs[2] = best_frenet_path->c_lateral_acceleration;
            fot_rv->costs[3] = best_frenet_path->c_lateral_jerk;
            fot_rv->costs[4] = best_frenet_path->c_lateral;
            fot_rv->costs[5] = best_frenet_path->c_longitudinal_acceleration;
            fot_rv->costs[6] = best_frenet_path->c_longitudinal_jerk;
            fot_rv->costs[7] = best_frenet_path->c_time_taken;
            fot_rv->costs[8] = best_frenet_path->c_end_speed_deviation;
            fot_rv->costs[9] = best_frenet_path->c_longitudinal;
            fot_rv->costs[10] = best_frenet_path->c_inv_dist_to_obstacles;
            fot_rv->costs[11] = best_frenet_path->cf;
        }
    }

    // Convert the initial conditions from cartesian space to frenet space
    void to_frenet_initial_conditions(
            double s0, double x, double y, double vx,
            double vy, double forward_speed, double* xp, double* yp, int np,
            double* initial_conditions
            ) {
        vector<double> wx (xp, xp + np);
        vector<double> wy (yp, yp + np);
        CubicSpline2D* csp = new CubicSpline2D(wx, wy);

        // get distance from car to spline and projection
        double s = csp->find_s(x, y, s0);
        double distance = norm(csp->calc_x(s) - x, csp->calc_y(s) - y);
        // Local patch: guard the exact-on-spline case (0/0 -> NaN c_d).
        double inv_distance = distance > 1e-9 ? 1.0 / distance : 0.0;
        tuple<double, double> bvec ((csp->calc_x(s) - x) * inv_distance,
                (csp->calc_y(s) - y) * inv_distance);

        // normal spline vector
        // Local patch: upstream evaluated this normal at the caller-supplied
        // s0 (the previous cycle's s, measured on a *different* local
        // spline), so in a corner the sign of c_d and c_d_d could flip and
        // the planned path started mirrored to the wrong side. Use the
        // tangent at the projection s found above (calc_yaw, rather than a
        // finite difference that can step past the spline end and get NaN).
        double yaw_s = csp->calc_yaw(s);
        double x0 = 0.0;
        double y0 = 0.0;
        double x1 = cos(yaw_s);
        double y1 = sin(yaw_s);

        // unit vector orthog. to spline
        tuple<double, double> tvec (y1-y0, -(x1-x0));
        as_unit_vector(tvec);

        // compute tangent / normal car vectors
        tuple<double, double> fvec (vx, vy);
        as_unit_vector(fvec);

        // get initial conditions in frenet frame
        initial_conditions[0] = s; // current longitudinal position s
        initial_conditions[1] = forward_speed; // speed [m/s]
        // lateral position c_d [m]
        initial_conditions[2] = copysign(distance, dot(tvec, bvec));
        // lateral speed c_d_d [m/s]
        initial_conditions[3] = -forward_speed * dot(tvec, fvec);
        initial_conditions[4] = 0.0; // lateral acceleration c_d_dd [m/s^2]
        // TODO: add lateral acceleration when CARLA 9.7 is patched (IMU)

        delete csp;
    }
}
