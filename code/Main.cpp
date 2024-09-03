#include "stdafx.h"
#include "tools.h"
#include "CNEllipseDetector.h"
#include <filesystem>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

// 全局变量设置
string INPUT_DIR = "/home/zihan/dataset/test-img/";
string OUTPUT_DIR = "/home/zihan/dataset/test-img/Jia/";
float fThScoreScore = 0.6f;
float fMinReliability = 0.4f;
float fTaoCenters = 0.04f;
int ThLength = 16;
float MinOrientedRectSide = 3.0f;

vector<double> ProcessImage(const string& filename)
{
    CNEllipseDetector cned;
    
    // 读取图片
    Mat3b image = imread(filename);
    if(image.empty()) {
        cout << "无法读取图片: " << filename << endl;
        return vector<double>();
    }
    
    Size sz = image.size();

    // 转换为灰度图
    Mat1b gray;
    cvtColor(image, gray, CV_BGR2GRAY);

    // 参数设置
    int iNs = 16;
    float fMaxCenterDistance = sqrt(float(sz.width*sz.width + sz.height*sz.height)) * fTaoCenters;
    Size szPreProcessingGaussKernelSize = Size(5,5);
    double dPreProcessingGaussSigma = 1.0;
    float fDistanceToEllipseContour = 0.1f;

    // 初始化检测器
    cned.SetParameters(
        szPreProcessingGaussKernelSize,
        dPreProcessingGaussSigma,
        1.0f,
        fMaxCenterDistance,
        ThLength,
        MinOrientedRectSide,
        fDistanceToEllipseContour,
        fThScoreScore,
        fMinReliability,
        iNs
    );

    // 检测椭圆
    vector<Ellipse> ellipses;
    Mat1b gray_clone = gray.clone();
    cned.Detect(gray_clone, ellipses);

    // 在图片上绘制检测到的椭圆
    Mat3b resultImage = image.clone();
    cned.DrawDetectedEllipses(resultImage, ellipses);

    // 保存结果图片
    fs::path input_path(filename);
    string output_filename = OUTPUT_DIR + input_path.filename().string();
    imwrite(output_filename, resultImage);

    // 返回处理时间等信息
    vector<double> times = cned.GetTimes();
    times.push_back(ellipses.size()); // 添加检测到的椭圆数量
    return times;
}

int main()
{
    // 确保输出目录存在
    fs::create_directories(OUTPUT_DIR);

    vector<string> resultString;
    resultString.push_back("Filename,Edge Detection,Pre processing,Grouping,Estimation,Validation,Clustering,Total Time,Ellipses Count");

    for (const auto & entry : fs::directory_iterator(INPUT_DIR))
    {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png")
        {
            string filename = entry.path().string();
            vector<double> results = ProcessImage(filename);

            if (!results.empty())
            {
                double total_time = accumulate(results.begin(), results.begin() + 6, 0.0);
                stringstream ss;
                ss << entry.path().filename().string() << "," 
                   << results[0] << "," << results[1] << "," << results[2] << "," 
                   << results[3] << "," << results[4] << "," << results[5] << "," 
                   << total_time << "," << results[6];
                resultString.push_back(ss.str());

                cout << "处理完成: " << filename << ", 总时间: " << total_time << "ms, 检测到 " << results[6] << " 个椭圆" << endl;
            }
        }
    }

    // 保存结果到CSV文件
    ofstream outFile(OUTPUT_DIR + "results.csv");
    for (const auto& line : resultString)
    {
        outFile << line << endl;
    }
    outFile.close();

    cout << "所有图片处理完成，结果保存在 " << OUTPUT_DIR << endl;
    return 0;
}