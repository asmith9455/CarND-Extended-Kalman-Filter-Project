#ifndef TOOLS_H_
#define TOOLS_H_

#include <vector>
#include "Eigen/Dense"

void validate(const Eigen::VectorXd& vec);

Eigen::MatrixXd CalculateJacobian(const Eigen::VectorXd& x_state);

Eigen::VectorXd CalculateRMSE(const std::vector<Eigen::VectorXd> &estimations, 
                                const std::vector<Eigen::VectorXd> &ground_truth);

#endif  // TOOLS_H_
