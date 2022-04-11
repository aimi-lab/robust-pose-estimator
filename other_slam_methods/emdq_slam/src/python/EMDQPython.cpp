#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API
#include <opencv2/core/core.hpp>
#include <pyboostcvconverter/pyboostcvconverter.hpp>
#include <EMDQ.h>
#include "EMDQPython.h"
#include <boost/python/stl_iterator.hpp>

#if (PY_VERSION_HEX >= 0x03000000)
static void* init_ar() {
#else
static void init_ar() {
#endif
    Py_Initialize();

    import_array();
    return NUMPY_IMPORT_ARRAY_RETVAL;
}

namespace py = boost::python;

template< typename T >
inline
std::vector< T > to_std_vector( const py::list& iterable )
{
    return std::vector< T >( py::stl_input_iterator< T >( iterable ),
                             py::stl_input_iterator< T >( ) );
}

BOOST_PYTHON_MODULE(emdq)
{
    init_ar();

    boost::python::to_python_converter<cv::Mat, pbcvt::matToNDArrayBoostConverter>();
    pbcvt::matFromNDArrayBoostConverter();

    boost::python::class_<emdqPython, boost::noncopyable>("pyEMDQ", boost::python::init<const double>())
        .def("SetScale", &emdqPython::SetScale)
        //.def("InputMatchingDataFromOpenCV", &emdqPython::InputMatchingDataFromOpenCV)
        .def("fit", &emdqPython::fit)
        .def("predict", &emdqPython::predict);
        //.def("GetNumberOfInliers", &emdqPython::GetNumberOfInliers)
        //.def("GetNumberOfTotalMatches", &emdqPython::GetNumberOfTotalMatches)
        //.def("GetDeformedCoord", &emdqPython::GetDeformedCoord)
        //.def("ComputeDeformationFieldAtGivenPosition", &emdqPython::ComputeDeformationFieldAtGivenPosition)
        //.def("GetDisplacedCoord", &emdqPython::GetDisplacedCoord);
}

emdqPython::emdqPython(double scale)
{
	emdq = std::make_shared<EMDQSLAM::CEMDQ>(scale);
}


emdqPython::~emdqPython()
{
	
}

void emdqPython::SetScale(double scale)
{
	emdq->SetScale(scale);
}


double emdqPython::fit(const cv::Mat& src_kpts, const cv::Mat& target_kpts, boost::python::list& matches)
{
    std::vector<cv::DMatch> matches_vec;
    for(int i=0; i < len(matches); i++)
    {
        boost::python::tuple m = boost::python::extract<boost::python::tuple>(matches[i]);
        cv::DMatch match = cv::DMatch(boost::python::extract<int>(m[0]),boost::python::extract<int>(m[1]), boost::python::extract<double>(m[2]));
        matches_vec.push_back(match);
    }
	emdq->InputMatchingDataFromOpenCV(src_kpts, target_kpts, &matches_vec);
	double n_inliers = emdq->Compute();
    return n_inliers;
}


cv::Mat emdqPython::predict(const cv::Mat& inputCoord)
{
	cv::Mat output(inputCoord.rows, 13, CV_64F);
	for (int i = 0; i < inputCoord.rows; i++)
    {
        const double* inputCoordi = inputCoord.ptr<double>(i);
        double* outputi = output.ptr<double>(i);
	    double coord1[3] = {(double)inputCoordi[0], (double)inputCoordi[1], (double)inputCoordi[2]};
	    double coord2[3] = {0.0, 0.0, 0.0};
	    double sigma2_change = 1.0;
	    double dq_change[8], mu_change;
	    emdq->ComputeDeformationFieldAtGivenPosition(coord1, coord2, sigma2_change, dq_change, mu_change);
	    outputi[0] = coord2[0];
	    outputi[1] = coord2[1];
        outputi[2] = coord2[2];
	    outputi[3] = sigma2_change;
	    outputi[4] = dq_change[0];
	    outputi[5] = dq_change[1];
	    outputi[6] = dq_change[2];
	    outputi[7] = dq_change[3];
        outputi[8] = dq_change[4];
	    outputi[9] = dq_change[5];
	    outputi[10] = dq_change[6];
	    outputi[11] = dq_change[7];
	    outputi[12] = mu_change;
	}
    return output;
}


