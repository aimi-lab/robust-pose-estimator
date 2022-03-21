#ifndef EMDQPYTHON_H
#define EMDQPYTHON_H

#include <memory>
#include <Python.h>
#include <boost/python.hpp>
#include <EMDQ.h>
#include <opencv2/core/core.hpp>

class emdqPython
{
public:
    emdqPython(double scale);
    ~emdqPython();
    
    void SetScale(double scale);
    double fit(const cv::Mat& src_kpts, const cv::Mat& target_kpts, boost::python::list& matches);
    cv::Mat predict(const cv::Mat& inputCoord);
    
private:
    std::shared_ptr<EMDQSLAM::CEMDQ> emdq;
};



#endif // EMDQPYTHON_H
