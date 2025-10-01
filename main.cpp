#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

// ---------------- Trajectory Model ----------------
struct TrajectoryResidual {
    TrajectoryResidual(double t, double x_obs, double y_obs)
        : t_(t), x_obs_(x_obs), y_obs_(y_obs) {}

    template <typename T>
    bool operator()(const T* const params, T* residuals) const {
        // params: [x0, y0, vx0, vy0, k, g]
        T x0 = params[0];
        T y0 = params[1];
        T vx0 = params[2];
        T vy0 = params[3];
        T k   = params[4];
        T g   = params[5];

        // 数学坐标系（y 向上）
        T x_pred = x0 + (vx0 / k) * (T(1.0) - ceres::exp(-k * T(t_)));
        T y_pred = y0 + ((vy0 + g / k) / k) * (T(1.0) - ceres::exp(-k * T(t_))) - (g / k) * T(t_);

        residuals[0] = x_pred - T(x_obs_);
        residuals[1] = y_pred - T(y_obs_);
        return true;
    }

private:
    double t_, x_obs_, y_obs_;
};

int main() {
    // 打开视频
    VideoCapture cap("video.mp4");
    if (!cap.isOpened()) {
        cerr << "无法打开视频文件！" << endl;
        return -1;
    }

    double fps = cap.get(CAP_PROP_FPS);

    Mat frame;
    vector<Point2f> observations;
    int frameIdx = 0;

    // ---------------- 目标检测 ----------------
    while (cap.read(frame)) {
        frameIdx++;

        Mat hsv, mask;
        cvtColor(frame, hsv, COLOR_BGR2HSV);

        // 蓝色范围
        Scalar lowerb(90, 50, 50);
        Scalar upperb(140, 255, 255);
        inRange(hsv, lowerb, upperb, mask);

        // 找轮廓
        vector<vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        if (!contours.empty()) {
            // 最大轮廓
            size_t largest = 0;
            double maxArea = 0;
            for (size_t i = 0; i < contours.size(); i++) {
                double area = contourArea(contours[i]);
                if (area > maxArea) {
                    maxArea = area;
                    largest = i;
                }
            }
            Moments mu = moments(contours[largest]);
            if (mu.m00 > 0) {
                Point2f center(mu.m10 / mu.m00, mu.m01 / mu.m00);
                // 转换到数学坐标系（y 向上）
                Point2f center_math(center.x, frame.rows - center.y);
                observations.push_back(center_math);
            }
        }
    }
    cap.release();

    if (observations.empty()) return 0;

    // ---------------- Ceres 拟合 ----------------
    double params[6] = {observations[0].x, observations[0].y, 100, 100, 0.066666, 500};

    ceres::Problem problem;
    for (size_t i = 0; i < observations.size(); i++) {
        double t = i / fps;
        ceres::CostFunction* cost =
            new ceres::AutoDiffCostFunction<TrajectoryResidual, 2, 6>(
                new TrajectoryResidual(t, observations[i].x, observations[i].y));
        problem.AddResidualBlock(cost, nullptr, params);
    }

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = false; // 关掉详细输出，加速
    options.linear_solver_type = ceres::DENSE_QR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 输出结果
    cout << summary.BriefReport() << endl;
    cout << "拟合参数: x0=" << params[0]
         << ", y0=" << params[1]
         << ", vx0=" << params[2]
         << ", vy0=" << params[3]
         << ", k=" << params[4]
         << ", g=" << params[5] << endl;

    return 0;
}
































