/************************************************************************/
// C++ code of the EMDQ algorithm
// <CopyRight 2020> Haoyin Zhou, Jagadeesan Jayender 
//
// Please cite our paper when using this code:
// Haoyin Zhou, Jagadeesan Jayender, "Smooth Deformation Field-based Mismatch Removal in Real-time"
//
// This program is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// General Public License for more details.       
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.
/************************************************************************/

#ifndef EMDQ_H
#define EMDQ_H

#include <iostream>
#include <vector>
#include <stdlib.h>
#include <stdio.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/flann/miniflann.hpp>

using namespace std;
using namespace cv;

namespace EMDQSLAM
{
	class CEMDQ
	{
	private: // parameters for EMDQ
		double scale;
		double inliersThreshold;	
		int LeastNumberOfInlierRANSACTrial;
		double beta;
		double beta2;
		int NeighborCount;
		double a;
		int EMMaxIterNumber;
		double pThreshold;
		double pdiffthreshold;
		
	private: // coordinates of the input matches
			 // the obtained deformation field makes matchUV1 = mu_matches dq_matches(matchUV2)
		std::vector< std::vector<double> > matchUV1;
		std::vector< std::vector<double> > matchUV2;

	private: // intermediate parameters used in the EMDQ algorithm
		std::vector< double > p_matches;
		std::vector< double > error2_matches;
		std::vector< std::vector<double> > dq_matches;
		std::vector< double > mu_matches;

	public: // mismatches removal results
		std::vector<bool> mask_R1PRNSC;
		std::vector<bool> mask_EMDQ;
		int numberOfEMDQInliers;
	
	public: // main functions
		CEMDQ(const double scale_in);
		~CEMDQ();

	public:	// input data
		void SetScale(const double scale_in); // aims to be self-adaptive to different image resolution
		bool InputMatchingDataFromOpenCV(const std::vector<KeyPoint>* keypoints1, const std::vector<KeyPoint>* keypoints2
							, const std::vector< cv::DMatch >*);
		bool InputMatchingDataFromOpenCV(const cv::Mat& coord1, const cv::Mat& coord2
			, const std::vector< cv::DMatch >*);
			
	public: // the main function of the EMDQ algorithm		
		int Compute();

	public: // IO functions
		int GetNumberOfInliers();
		int GetNumberOfTotalMatches();

	public: // IO functions, get the deformation field
		bool GetDeformedCoord(const double* x_in, double* x_out);
		bool ComputeDeformationFieldAtGivenPosition(const double* x_in, double* x_out							
								, double& sigma2_change_out, double* dq_change_out, double& mu_change_out);

	public: // visualization for demo, should be put someplace else ...
		void VisulizeResults(const std::vector<bool> mask, const cv::Mat& inputMat1, cv::Mat& outputMat1, const cv::Mat& inputMat2, cv::Mat& outputMat2);
		void Visulize2DDeformationField(const cv::Mat& inputMat2, cv::Mat& outputMat2
										, const int step);
	
	public: // temp, no dq used deformation
		bool GetDisplacedCoord(const double* x_in, double* x_out);
		
		void Visulize2DDisplacementInterplation(const cv::Mat& inputMat2, cv::Mat& outputMat2
										, const int step);								
	};
};

#endif // !EMDQ_H
