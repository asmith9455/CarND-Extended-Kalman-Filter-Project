#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF()
{
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  // Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  // R_laser_ << 0.0225, 0,
  //     0, 0.0225;
  R_laser_ << 0.0076, 0,
      0, 0.0078;

  //measurement covariance matrix - radar
  R_radar_ << 0.056, 0, 0,
      0, 0.116, 0,
      0, 0, 0.04;

  /**
   * TODO: Finish initializing the FusionEKF.
   * TODO: Set the process and measurement noises
   */
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

VectorXd state_to_measurement_radar(const VectorXd &x)
{
  VectorXd z_m(3);

  const double
      px = x(0),
      py = x(1),
      vx = x(2),
      vy = x(3);

  const double r = std::sqrt(px * px + py * py);

  z_m(0) = r;                  //range (distance from origin)
  z_m(1) = std::atan2(px, py); //angle about the z axis (0 at +x axis)

  if (r > 1e-10)
  {
    z_m(2) = (px * vx + py * vy) / r; //range rate
  }
  else
  {
    // the range rate is defined using the axis drawn between the radar and the target. when this distance is 0, the axis is undefined.
    // in reality, this doesn't seem to make much sense - how can the radar and the object occupy the same space??
    // we can, however, aim to estimate the range rate given the object's current velocity, which will enable robustness for testing
    // purposes, and perhaps also account for inaccurate measurements
    // for example, if the speed is 0, we expect the 'range rate' to be 0.

    if (std::sqrt(vx * vx + vy * vy) < 1e-10)
    {
      z_m(2) = 0.0;
    }

    // if the velocity is non \vec{0}, we need to do something more sophisticated.
    // we can estimate the next axis or the previous axis, and perform the projection
    // using this axis.
    // with a linear projection, the amount of 'time' we project into the future doesn't much
    // matter - just use 1 second to make it easier
    // therefore, just use the speed as an approximation of the range rate at this time
    // since the object is moving and has just passed over the radar, it's range rate will
    // jump from being highly negative to being highly positive (ex. -10 to +10).
    // this is a problem...
    else
    {
      z_m(2) = std::sqrt(vx * vx + vy * vy);
    }
  }

  return z_m;
}

void FusionEKF::InitializeFromMeasurement(const MeasurementPackage &measurement_pack)
{
  VectorXd x(4);
  x << 0, 0, 0, 0;
  MatrixXd P(4, 4);
  P << 10, 0, 0, 0,
      0, 10, 0, 0,
      0, 0, 100, 0,
      0, 0, 0, 100;

  MatrixXd F(4, 4);

  const double dt = 0.02;

  F << 1, 0, dt, 0,
      0, 1, 0, dt,
      0, 0, 1, 0,
      0, 0, 0, 1;

  MatrixXd H(1, 1), R(1, 1); //these will be initialized later depending on which sensor comes in

  H << 0;
  R << 0;

  MatrixXd Q(4, 4);

  Q << 0.0, 0, 0, 0,
      0, 0.0, 0, 0,
      0, 0, 0.0, 0,
      0, 0, 0, 0.0;

  ekf_.Init(x, P, F, H, R, Q);

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR)
  {
    const double
        range = measurement_pack.raw_measurements_(0),
        z_angle = measurement_pack.raw_measurements_(1);
    //range_rate = measurement_pack.raw_measurements_(2); //is there a way to use the range rate to initialize the speed of the KF?

    ekf_.x_(0) = range * std::cos(z_angle);
    ekf_.x_(1) = range * std::sin(z_angle);
  }
  else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER)
  {
    const double
        px = measurement_pack.raw_measurements_(0),
        py = measurement_pack.raw_measurements_(1);

    ekf_.x_(0) = px;
    ekf_.x_(1) = py;
  }
}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack)
{
  /**
   * Initialization
   */
  if (!is_initialized_)
  {
    InitializeFromMeasurement(measurement_pack);
    is_initialized_ = true;
  }
  else
  {
    Predict(measurement_pack.timestamp_);
    Update(measurement_pack);
  }

  PrintDebugInfo(measurement_pack);

  previous_timestamp_ = measurement_pack.timestamp_;

  validate(ekf_.x_);

  std::cout << "-------------------------------------------------------------------" << std::endl;
  std::cout << "-------------------------------------------------------------------" << std::endl;
}

void FusionEKF::Predict(long long new_timestamp)
{

  //params for q matrix
  const double var_ax = 9e0; //noise_ax = std_dev_ax * std_dev_ax
  const double var_ay = 9e0;

  //1e-5 //acceleration not modelled well enough
  //...
  //1e0 // works ok but not great (.1786, 0.1527, .7320, .6670 )
  //5e0 (0.1211, 0.1032, 0.5957, 0.4900)
  //9e0 (0.1153, 0.1023, 0.5832, 0.4687)
  //solution
  //1e1 (0.1148, 0.1027 0.5833, 0.4679) BEST SO FAR
  //2e1 (.1144, .1076, .6036, .4881)
  //1e2 // can't even see estimates in the simulator

  //using the true values of the standard deviation of ax, ay doesn't work well.

  MatrixXd Q(4, 4);

  if (new_timestamp < previous_timestamp_)
  {
    throw std::runtime_error("Expected monotonic increase of timestamps in sequential measurements.");
  }

  const double
      dt = static_cast<double>(new_timestamp - previous_timestamp_) * 1e-6,
      dt2 = dt * dt,
      dt3 = dt2 * dt,
      dt4 = dt3 * dt;

  Q << dt4 * 0.25 * var_ax, 0, dt3 * 0.5 * var_ax, 0,
      0, dt4 * 0.25 * var_ay, 0, dt3 * 0.5 * var_ay,
      dt3 * 0.5 * var_ax, 0, dt2 * var_ax, 0,
      0, dt3 * 0.5 * var_ay, 0, dt2 * var_ay;

  MatrixXd F(4, 4);

  F << 1, 0, dt, 0,
      0, 1, 0, dt,
      0, 0, 1, 0,
      0, 0, 0, 1;

  ekf_.F_ = F;
  ekf_.Q_ = Q;

  ekf_.Predict();

  std::cout << "\n\n\n-------------------------------------------------------------------" << std::endl;
  std::cout << "-------------------------------------------------------------------" << std::endl;
  std::cout << "after prediction (time difference is: " << dt << " seconds): " << std::endl;
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;

  validate(ekf_.x_);
}

void FusionEKF::Update(const MeasurementPackage &measurement_pack)
{
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR)
  {
    // static int limiter = 0;

    // if (limiter > 10)
    //   return;

    // ++limiter;

    bool use_EKF = true;

    if (use_EKF)
    {
      ekf_.R_ = R_radar_;

      auto calc_jacobian = [](Eigen::VectorXd state) { return CalculateJacobian(state); };
      auto h = [](Eigen::VectorXd state) { return (state_to_measurement_radar(state)); };
      ekf_.UpdateEKF(measurement_pack.raw_measurements_, h, calc_jacobian);
    }
    else
    {
      ekf_.R_ = R_radar_.block(0, 0, 2, 2);
      MatrixXd H(2, 4);

      H << 1, 0, 0, 0,
          0, 1, 0, 0;

      ekf_.H_ = H;

      VectorXd pseudo_z(2);

      pseudo_z(0) = measurement_pack.raw_measurements_(0) * std::cos(measurement_pack.raw_measurements_(1));
      pseudo_z(1) = measurement_pack.raw_measurements_(0) * std::sin(measurement_pack.raw_measurements_(1));

      ekf_.Update(pseudo_z);
    }


    validate(ekf_.x_);

    std::cout << "-------------------------------------------------------------------" << std::endl;
    std::cout << "after measurement update with RADAR:" << std::endl;
    // std::cout << "limiter:" << limiter << std::endl;
    cout << "x_ = " << ekf_.x_ << endl;
    cout << "P_ = " << ekf_.P_ << endl;
  }
  else
  {
    MatrixXd H(2, 4);

    H << 1, 0, 0, 0,
        0, 1, 0, 0;

    ekf_.H_ = H;
    ekf_.R_ = R_laser_;

    ekf_.Update(measurement_pack.raw_measurements_);

    std::cout << "-------------------------------------------------------------------" << std::endl;
    std::cout << "after measurement update with LIDAR:" << std::endl;
    cout << "x_ = " << ekf_.x_ << endl;
    cout << "P_ = " << ekf_.P_ << endl;
  }
}

void FusionEKF::PrintDebugInfo(const MeasurementPackage &measurement_pack)
{
}
