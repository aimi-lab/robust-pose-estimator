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


#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>

#include "EMDQ.h"

using namespace std;
using namespace cv::xfeatures2d;
using namespace cv;


/*
./demoEMDQ /home/slam-lab/work/Data/MPI_Sintel/training/clean/mountain_1/frame_0001.png /home/slam-lab/work/Data/MPI_Sintel/training/clean/mountain_1/frame_0012.png 

./demoEMDQ /home/slam-lab/work/Video2ImagesFolder/data/imgs/img_00001.jpg /home/slam-lab/work/Video2ImagesFolder/data/imgs/img_00312.jpg

./demoEMDQ /home/slam-lab/work/Data/KITTI/2011_09_26/2011_09_26_drive_0002_sync/image_00/data/0000000000.png /home/slam-lab/work/Data/KITTI/2011_09_26/2011_09_26_drive_0002_sync/image_00/data/0000000010.png 

 ./demoEMDQ /home/slam-lab/work/Data/KITTI/2011_09_26/2011_09_26_drive_0011_sync/image_02/data/0000000000.png /home/slam-lab/work/Data/KITTI/2011_09_26/2011_09_26_drive_0011_sync/image_02/data/0000000005.png 
 
  ./demoEMDQ /home/slam-lab/work/Data/KITTI/2011_09_26/2011_09_26_drive_0011_sync/image_02/data/0000000000.png /home/slam-lab/work/Data/KITTI/2011_09_26/2011_09_26_drive_0011_sync/image_02/data/0000000009.png   (have a look)
*/


int main(int argc, char** argv)
{
	if( argc != 3)
	{
		std::cout <<" Usage: ./demoEMDQ PathToImage1 PathToImage2" << std::endl;
		std::cout <<" Example: ./demoEMDQ picture1.png picture2.png" << std::endl;
		return -1;
	}
	
	/**************************** Read two images *****************************/
    Mat img1, img2;
	img1 = imread(argv[1]);   // Read the file
 	img2 = imread(argv[2]);   // Read the file
    if(! img1.data || ! img2.data )   // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

	/***************** Standard OpenCV-based SURF Matching ********************/
	std::vector<KeyPoint> keypoints1, keypoints2;
	cv::Mat descriptors1, descriptors2;
	std::vector< cv::DMatch > matches;
	{
		int minHessian = 10;
		Ptr<SURF> detector = SURF::create(minHessian);
		detector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
		detector->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

		std::vector< cv::DMatch > matches_raw;
		Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
		std::vector< std::vector<DMatch> > knn_matches;
		matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);

		const float ratio_thresh = 0.7f;
		for (size_t i = 0; i < knn_matches.size(); i++)
		{
			if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
			{
				matches.push_back(knn_matches[i][0]);
			}
		}
	}

	/********************* Initial EMDQ Algorithm *************************************/
	// scale aims to make EMDQ self-adaptive to different image resolutions, and 480x270 is used as the reference resolution
	const double scale = 0.5 * ((double)img1.cols / 480.0 + (double)img1.rows / 270.0); 
	EMDQSLAM::CEMDQ EMDQAlgorithm(scale);
	
	/********************* Input OpenCV results to EMDQ *******************************/
	// please note that we overloaded the input function to make it easier to use
	// one takes std::vector<cv::KeyPoint> as the input, another one takes  std::vector< std::vector<double> >.
	if (EMDQAlgorithm.InputMatchingDataFromOpenCV(&keypoints1, &keypoints2, &matches) == false)
	{
		std::cerr << "InputMatchingDataFromOpenCV fails..." << std::endl;
		return -1;
	}
	std::cout << "number of input feature matches = " << EMDQAlgorithm.GetNumberOfTotalMatches() << std::endl;

	/********************* EMDQ main function ****************************************/
	EMDQAlgorithm.Compute();
	
	/************************ mismatch removal results *******************************/
	std::cout << "number of inliers = " << EMDQAlgorithm.GetNumberOfInliers() << std::endl;

	// std::vector<bool> EMDQAlgorithm.mask_EMDQ is the label list of inliers (true) or outliers (false). 
	// std::vector<bool> EMDQAlgorithm.mask_R1PRNSC is the results of R1P-RNSC only
	Mat img1_misMatchRemoval;
	Mat img2_misMatchRemoval;
	EMDQAlgorithm.VisulizeResults(EMDQAlgorithm.mask_EMDQ, img1, img1_misMatchRemoval, img2, img2_misMatchRemoval); // a simple visualization function 
	imshow("mismatches removal results 1", img1_misMatchRemoval);                  
	imshow("mismatches removal results 2", img2_misMatchRemoval);                  

	/************************* Deformation Field *************************************/
	cv::Mat frame_deformationField;		
	const int step = 40; // the step to draw the arrows (in pixels)
	EMDQAlgorithm.Visulize2DDeformationField(img2, frame_deformationField, step);
	imshow("deformation field", frame_deformationField);		

	/************************* temp, displacment from coordinate interplation, no dq is used *************************************/
	cv::Mat frame_displacment;		
	EMDQAlgorithm.Visulize2DDisplacementInterplation(img2, frame_displacment, step);
	imshow("frame_displacment (temp)", frame_displacment);		


	/******* Examples of how to move a pixel according to the deformation field************/
	{ // if only need the coordinates
		double coord1[2] = {100.0, 80.0};
		double coord2[2] = {0.0, 0.0};
		EMDQAlgorithm.GetDeformedCoord(coord1, coord2);
		std::cout << "example 1: pixel (" << coord1[0] << ", " << coord1[1] << ") is moved to (" << coord2[0] << ", " << coord2[1] << ")" << std::endl;
	}
	{ // if need all information of the deformation field
		double coord1[2] = {100.0, 80.0};
		double coord2[2] = {0.0, 0.0};
		double sigma2_change = 1.0;
		double dq_change[4], mu_change;
		EMDQAlgorithm.ComputeDeformationFieldAtGivenPosition(coord1, coord2
					, sigma2_change, dq_change, mu_change);
					
		std::cout << "example 2: pixel (" << coord1[0] << ", " << coord1[1] << ") is moved to (" << coord2[0] << ", " << coord2[1] << ")" << std::endl;
		std::cout << "example 2: related dual quaternion is " << dq_change[0] << ", " << dq_change[1] << ", " << dq_change[2] << ", " << dq_change[3] << ", scale is " << mu_change << ", uncertainty sigma2 = " << sigma2_change << std::endl;
		
	}	
	
	/*************************** Press any key to end *********************************/
    waitKey(0);
    return 0;
}
