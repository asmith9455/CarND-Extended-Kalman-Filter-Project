#include "kalman_filter.h"
#include <cmath>
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {

  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;

}

void KalmanFilter::Update(const VectorXd &z) {
  
  const MatrixXd y = z - H_ * x_;
  const MatrixXd S = H_ * P_ * H_.transpose() + R_;
  const MatrixXd K = P_ * H_.transpose() * S.inverse();

  x_ = x_ + K * y;
  P_ = (Eigen::MatrixXd::Identity(4,4) - K * H_) * P_;

}

void KalmanFilter::UpdateEKF(const VectorXd &z, std::function<VectorXd(VectorXd)> h, std::function<MatrixXd(VectorXd)> calc_jacobian) {
  
  const MatrixXd y = z - h(x_);

  //compute Hj (jacobian)

  const MatrixXd Hj = calc_jacobian(x_);

  const MatrixXd S = Hj * P_ * Hj.transpose() + R_;
  const MatrixXd K = P_ * Hj.transpose() * S.inverse();

  x_ = x_ + K * y;
  P_ = (Eigen::MatrixXd::Identity(4,4) - K * Hj) * P_;

}
