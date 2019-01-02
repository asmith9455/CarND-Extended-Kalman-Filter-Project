#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

VectorXd CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  if (estimations.empty())
  {
    throw std::runtime_error("No estimations passed for calculation of rmse - the number of estimations must match the number of ground truth values.");
  }

  if (estimations.size() != ground_truth.size())
  {
    throw std::runtime_error("Incorrect number of estimations passed for calculation of rmse - the number of estimations must match the number of ground truth values.");
  }

  for (size_t i=0; i < estimations.size(); ++i) 
  {
    rmse += (estimations[i] - ground_truth[i]).array().pow(2).matrix();
  }
  
  rmse /= static_cast<double>(estimations.size());
  
  rmse = rmse.array().sqrt().matrix();
  
  return rmse;
}

MatrixXd CalculateJacobian(const VectorXd& x_state) {
  
  MatrixXd Hj(3, 4);

  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);
  
  if (std::fabs(px) > 1e-10 || std::fabs(py) > 1e-10)
  {
    Hj << px / std::sqrt(px*px + py*py), py / std::sqrt(px*px + py*py), 0, 0,
            -py / (px*px + py*py), px / (px*px + py*py), 0, 0,
            py*(vx*py-vy*px)/std::pow(px*px+py*py, 1.5), px*(vy*px-vx*py)/std::pow(px*px+py*py,1.5), px/std::sqrt(px*px+py*py), py/std::sqrt(px*px+py*py);

    // Hj << 0, 0, 0, 0,
    //         0, 0, 0, 0,
    //         0, 0, 0, 0;
  }
  else
  {
    Hj << 0, 0, 0, 0,
          0, 0, 0, 0,
          0, 0, 0, 0; //shouldn't do this... it prevents the information from the measurement from being absorbed...

    //when the state is entirely 0, we get trapped - the jacobian is 0 and therefore K is 0 - then the state remains 0 and we are stuck
  }

  return Hj;
}

void validate(const Eigen::VectorXd& vec)
{
  for (size_t i = 0; i < vec.rows(); ++i)
  {
    if (std::isnan(vec(i)))
    {
      throw std::runtime_error("unexpected state value");
    }
  }
}
