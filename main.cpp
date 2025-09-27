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

        // 拟合时用数学坐标系（y 向上）
        T x_pred = x0 + (vx0 / k) * (T(1.0) - ceres::exp(-k * T(t_)));
        T y_pred = y0 + ((vy0 + g / k) / k) * (T(1.0) - ceres::exp(-k * T(t_))) - (g / k) * T(t_);

        residuals[0] = x_pred - T(x_obs_);
        residuals[1] = y_pred - T(y_obs_);
        return true;
    }

private:
    double t_, x_obs_, y_obs_;
};

// 计算轨迹点（数学坐标系）
Point2f trajectory(double t,
                   double x0, double y0,
                   double vx0, double vy0,
                   double k, double g)
{
    double x = x0 + (vx0 / k) * (1.0 - exp(-k * t));
    double y = y0 + ((vy0 + g / k) / k) * (1.0 - exp(-k * t)) - (g / k) * t;
    return Point2f((float)x, (float)y);
}

// 绘制拟合曲线（注意要翻回图像坐标）
void drawTrajectory(Mat& frame,
                    double x0, double y0,
                    double vx0, double vy0,
                    double k, double g,
                    double fps,
                    int totalFrames)
{
    vector<Point2f> traj_points;
    for (int i = 0; i < totalFrames; i++) {
        double t = i / fps;
        Point2f p = trajectory(t, x0, y0, vx0, vy0, k, g);
        // 翻回 OpenCV 图像坐标
        p.y = frame.rows - p.y;
        traj_points.push_back(p);
    }

    for (size_t i = 1; i < traj_points.size(); i++) {
        line(frame, traj_points[i-1], traj_points[i], Scalar(255, 0, 0), 2);
    }
}

// 对观测点进行滑动平均平滑
vector<Point2f> smoothObservations(const vector<Point2f>& observations, int win=3) {
    vector<Point2f> smoothed;
    for (int i = 0; i < observations.size(); i++) {
        Point2f avg(0,0);
        int cnt = 0;
        for (int j = max(0, i-win); j <= min((int)observations.size()-1, i+win); j++) {
            avg += observations[j];
            cnt++;
        }
        avg.x /= cnt; avg.y /= cnt;
        smoothed.push_back(avg);
    }
    return smoothed;
}

int main() {
    // 打开视频
    VideoCapture cap("video.mp4");
    if (!cap.isOpened()) {
        cerr << "无法打开视频文件！" << endl;
        return -1;
    }

    double fps = cap.get(CAP_PROP_FPS);
    cout << "FPS = " << fps << endl;

    Mat frame;
    vector<Point2f> observations;
    int frameIdx = 0;

    // ---------------- 目标检测 ----------------
    while (cap.read(frame)) {
        frameIdx++;

        Mat hsv, mask;
        cvtColor(frame, hsv, COLOR_BGR2HSV);

        // 蓝色范围（根据实际需要调整）
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

                // 显示时仍然用原始坐标
                circle(frame, center, 4, Scalar(0, 0, 255), -1); // 红色点
            }
        }

        imshow("Detection", frame);
        if (waitKey(1) == 27) break; // ESC退出
    }
    cap.release();
    destroyAllWindows();

    cout << "原始观测点数量 = " << observations.size() << endl;
    if (observations.empty()) return 0;

    // 平滑观测点
    observations = smoothObservations(observations, 3);
    cout << "平滑后点数量 = " << observations.size() << endl;

    // ---------------- Ceres 拟合 ----------------
    // 初始参数 (可用更合理的近似值)
    double params[6] = {observations[0].x, observations[0].y, 100, 100, 0.05, 500};

    ceres::Problem problem;
    for (size_t i = 0; i < observations.size(); i++) {
        double t = i / fps;
        ceres::CostFunction* cost =
            new ceres::AutoDiffCostFunction<TrajectoryResidual, 2, 6>(
                new TrajectoryResidual(t, observations[i].x, observations[i].y));
        problem.AddResidualBlock(cost, nullptr, params);
    }

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.linear_solver_type = ceres::DENSE_QR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    cout << summary.BriefReport() << endl;
    cout << "拟合参数: x0=" << params[0]
         << ", y0=" << params[1]
         << ", vx0=" << params[2]
         << ", vy0=" << params[3]
         << ", k=" << params[4]
         << ", g=" << params[5] << endl;

    // ---------------- 绘制拟合轨迹 ----------------
    cap.open("video.mp4");
    frameIdx = 0;

    while (cap.read(frame)) {
        frameIdx++;

        // 画轨迹曲线（用最终拟合的参数）
        drawTrajectory(frame, params[0], params[1], params[2], params[3], params[4], params[5], fps, observations.size());

        // 画观测点（翻回图像坐标）
        if (frameIdx <= observations.size()) {
            Point2f obs = observations[frameIdx-1];
            obs.y = frame.rows - obs.y;
            circle(frame, obs, 3, Scalar(0, 0, 255), -1);
        }

        // 显示参数
        string text = "x0=" + to_string((int)params[0]) +
                      " y0=" + to_string((int)params[1]) +
                      " vx0=" + to_string((int)params[2]) +
                      " vy0=" + to_string((int)params[3]) +
                      " k=" + to_string(params[4]) +
                      " g=" + to_string(params[5]);
        putText(frame, text, Point(30, 30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0,255,0), 2);

        imshow("Trajectory Fitting", frame);
        if (waitKey(30) == 27) break;
    }

    return 0;
}
































