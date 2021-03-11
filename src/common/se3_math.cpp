#include <Eigen/Dense>

void vee_map_3x3(Eigen::Matrix3f& mat, Eigen::Vector3f& vec)
{
        vec(0) = mat(2, 1);
        vec(1) = mat(0, 2);
        vec(2) = mat(1, 0);
}

void hat_map_3x3(Eigen::Vector3f& vec, Eigen::Matrix3f& mat)
{
        mat(0, 0) = 0.0f;
        mat(0, 1) = -vec(2);
        mat(0, 2) = +vec(1);
        mat(1, 0) = +vec(2);
        mat(1, 1) = 0.0f;
        mat(1, 2) = -vec(0);
        mat(2, 0) = -vec(1);
        mat(2, 1) = +vec(0);
        mat(2, 2) = 0.0f;
}
