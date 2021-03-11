#pragma once

#include <Eigen/Dense>

void vee_map_3x3(Eigen::Matrix3f& mat, Eigen::Vector3f& vec);
void hat_map_3x3(Eigen::Vector3f& vec, Eigen::Matrix3f& mat);
