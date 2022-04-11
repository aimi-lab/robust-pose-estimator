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


#include "EMDQ.h"
#include "DualQ.h"

#include <Eigen/Dense>

using namespace Eigen;
using Eigen::MatrixXd;
using Eigen::MatrixXi;


void GenerateDQfromEigenRt(const MatrixXd& R, const MatrixXd& t
	, double* dq, int D);

void GenerateWeightMatrixUsingSort(const MatrixXd& X, MatrixXd& K, MatrixXi& neighborIds, const int M);

void GenerateKaw(const MatrixXd& X1, const MatrixXd& X2, const int M
	, MatrixXd& Kraw, MatrixXi& neighborIds
	, const double beta_in);

void GenerateNeighborMatrixFromVector(const MatrixXd& p, const MatrixXi& neighborIds
	, MatrixXd& p_neighbor);

void WarpEigenPointsFromDQ(const MatrixXd& X_in, const MatrixXd& dq_in
	, MatrixXd& X_out);

EMDQSLAM::CEMDQ::CEMDQ(const double scale_in)
{
	this->numberOfEMDQInliers = 0;
	
	scale = scale_in;//1.0;
	inliersThreshold = 20.0 * scale_in;
	LeastNumberOfInlierRANSACTrial = 15;
	beta = 1.0 / 6400.0 / scale_in / scale_in;
	beta2 = 0.001 / scale_in / scale_in;
	NeighborCount = 50;
	a = 1e-5 / scale_in / scale_in;
	EMMaxIterNumber = 15;
	pThreshold = 0.5;
	pdiffthreshold = 0.005;
}

void EMDQSLAM::CEMDQ::SetScale(const double scale_in)
{
	this->scale = scale_in;
	
	inliersThreshold = 20.0 * scale;
	beta = 1.0 / 6400.0 / scale / scale;
	beta2 = 0.001 / scale_in / scale_in;
	a = 1e-5 / scale / scale;
}


EMDQSLAM::CEMDQ::~CEMDQ()
{
	for (size_t i = 0; i < this->matchUV1.size(); i++)
		this->matchUV1.at(i).clear();
	this->matchUV1.clear();

	for (size_t i = 0; i < this->matchUV2.size(); i++)
		this->matchUV2.at(i).clear();
	this->matchUV2.clear();

	this->p_matches.clear();
	this->error2_matches.clear();
	for (size_t i = 0; i < this->dq_matches.size(); i++)
		this->dq_matches.at(i).clear();
	this->dq_matches.clear();

	this->mu_matches.clear();
	this->mask_R1PRNSC.clear();
	this->mask_EMDQ.clear();
}



bool EMDQSLAM::CEMDQ::InputMatchingDataFromOpenCV(const std::vector<cv::KeyPoint>* keypoints1, const std::vector<cv::KeyPoint>* keypoints2
	, const std::vector< cv::DMatch >* matches)
{
	if (matches->size() == 0) return false;

	for (size_t i = 0; i < this->matchUV1.size(); i++)
		this->matchUV1.at(i).clear();
	this->matchUV1.clear();
	for (size_t i = 0; i < this->matchUV2.size(); i++)
		this->matchUV2.at(i).clear();
	this->matchUV2.clear();

	this->matchUV1.reserve(matches->size());
	this->matchUV2.reserve(matches->size());
	for (size_t i = 0; i < matches->size(); i++)
	{
		size_t pid1 = matches->at(i).queryIdx;
		size_t pid2 = matches->at(i).trainIdx;

		std::vector<double> matchUV1_this;
		matchUV1_this.push_back(keypoints1->at(pid1).pt.x);
		matchUV1_this.push_back(keypoints1->at(pid1).pt.y);
		this->matchUV1.emplace_back(matchUV1_this);

		std::vector<double> matchUV2_this;
		matchUV2_this.push_back(keypoints2->at(pid2).pt.x);
		matchUV2_this.push_back(keypoints2->at(pid2).pt.y);
		this->matchUV2.emplace_back(matchUV2_this);
	}

	this->p_matches.reserve(matches->size());
	this->error2_matches.reserve(matches->size());
	this->dq_matches.reserve(matches->size());
	this->mu_matches.reserve(matches->size());
	this->mask_R1PRNSC.reserve(matches->size());
	this->mask_EMDQ.reserve(matches->size());
	this->numberOfEMDQInliers = 0;
	return true;
}

bool EMDQSLAM::CEMDQ::InputMatchingDataFromOpenCV(const cv::Mat& coord1, const cv::Mat& coord2
	, const std::vector< cv::DMatch >* matches)
{
	if (matches->size() == 0) return false;
	int D = coord1.cols;

	for (size_t i = 0; i < this->matchUV1.size(); i++)
		this->matchUV1.at(i).clear();
	this->matchUV1.clear();
	for (size_t i = 0; i < this->matchUV2.size(); i++)
		this->matchUV2.at(i).clear();
	this->matchUV2.clear();

	this->matchUV1.reserve(matches->size());
	this->matchUV2.reserve(matches->size());
	for (size_t i = 0; i < matches->size(); i++)
	{
		size_t pid1 = matches->at(i).queryIdx;
		size_t pid2 = matches->at(i).trainIdx;

		std::vector<double> matchUV1_this;
		for (int l = 0; l < D; l ++)
			//matchUV1_this.push_back(coord1->at(pid1).at(l));
			matchUV1_this.push_back(coord1.at<double>(pid1, l));
		this->matchUV1.push_back(matchUV1_this);

		std::vector<double> matchUV2_this;
		for (int l = 0; l < D; l++)
			//matchUV2_this.push_back(coord2->at(pid2).at(l));
			matchUV2_this.push_back(coord2.at<double>(pid2, l));
		this->matchUV2.push_back(matchUV2_this);
	}

	this->p_matches.reserve(matches->size());
	this->error2_matches.reserve(matches->size());
	this->dq_matches.reserve(matches->size());
	this->mu_matches.reserve(matches->size());
	this->mask_R1PRNSC.reserve(matches->size());
	this->mask_EMDQ.reserve(matches->size());
	this->numberOfEMDQInliers = 0;
	return true;
}


int EMDQSLAM::CEMDQ::Compute()
{
	if (this->matchUV1.size() == 0 || this->matchUV2.size() == 0)
	{
		std::cerr << "EMDQ: no data!" << std::endl;
		return 0;
	}
	if (this->matchUV1.size() != this->matchUV2.size())
	{
		std::cerr << "EMDQ: sizes are different!" << std::endl;
		return 0;
	}
	this->mask_R1PRNSC.clear();
	this->mask_EMDQ.clear();
	for (size_t i = 0; i < this->dq_matches.size(); i++) this->dq_matches.at(i).clear();
	this->dq_matches.clear();
	this->mu_matches.clear();

	int N = (int)matchUV1.size();
	int D = (int)matchUV1.at(0).size();
	for (int i = 0; i < N; i++) this->mask_R1PRNSC.push_back(false);
	for (int i = 0; i < N; i++) this->mask_EMDQ.push_back(false);

	int dqsize = (D == 2) ? 4 : 8;
	MatrixXd X1(D, N), X2(D, N);
	for (int n = 0; n < N; n++)
	{
		for (int d = 0; d < D; d++) X1(d, n) = matchUV1.at(n).at(d);
		for (int d = 0; d < D; d++) X2(d, n) = matchUV2.at(n).at(d);
	}
	MatrixXd dq_points = MatrixXd::Zero(N, dqsize);
	dq_points.col(0).setOnes();
	MatrixXd mu_points = MatrixXd::Ones(N, 1);

	// R1P-RNSC
	std::vector<int> ransactryid;
	for (int i = 0; i < N; i++) ransactryid.push_back(i);
	cv::randShuffle(ransactryid);
	//	std::vector<bool> inliersMask_R1PRNSC;
	//	for (size_t i = 0; i < N; i++) inliersMask_R1PRNSC.push_back(false);
	MatrixXd distance_all(1, N);
	for (int i = 0; i < N; i++) distance_all(0, i) = 1e+8;
	MatrixXd SumRNSCWeight = MatrixXd::Ones(1, N);
	double punish = 1e+6;
	double inliersRatio = 0.0;
	int MaxTry = ((int)N > 100) ? 100 : (int)N;
	for (int ransactry = 0; ransactry < MaxTry; ransactry++)
	{
		size_t benchPid = ransactryid.at(ransactry);
		if (this->mask_R1PRNSC.at(benchPid) == true) continue;
		if (punish < (double)ransactry) break;
	//	std::cout << "benchPid = " << benchPid << std::endl;

		MatrixXd X1_nobench(D, N - 1), X2_nobench(D, N - 1);
		X1_nobench.block(0, 0, D, benchPid) = X1.block(0, 0, D, benchPid);
		X1_nobench.block(0, benchPid, D, N - benchPid - 1) = X1.block(0, benchPid + 1, D, N - benchPid - 1);
		X2_nobench.block(0, 0, D, benchPid) = X2.block(0, 0, D, benchPid);
		X2_nobench.block(0, benchPid, D, N - benchPid - 1) = X2.block(0, benchPid + 1, D, N - benchPid - 1);

		MatrixXd S1(D, N - 1), S2(D, N - 1);
		S1 = X1_nobench - X1.col(benchPid).replicate(1, N - 1);
		S2 = X2_nobench - X2.col(benchPid).replicate(1, N - 1);
		//	std::cout << "S1 = " << S1 << std::endl;
		//	std::cout << "S2 = " << S2 << std::endl;

		MatrixXd R(D, D); R.setIdentity();
		double mu = 1.0;
		//	std::cout << "R = " << R << std::endl;
		MatrixXd distance(1, N - 1);
		for (int iter = 0; iter < 3; iter++)
		{
			MatrixXd RS2 = mu * R * S2;
			//	std::cout << "RS2 = " << RS2 << std::endl;
			MatrixXd S2RS2 = S1 - RS2;
			//	std::cout << "S2RS2 = " << S2RS2 << std::endl;
			distance = S2RS2.colwise().norm();
			//	std::cout << "distance = " << distance << std::endl;
			MatrixXd p_this(1, N - 1);
			p_this = distance.array().inverse() * inliersThreshold;
			for (int i = 0; i < N - 1; i++)
			{
				if (p_this(i) > 1.0) p_this(0, i) = 1.0;
				if (p_this(i) < 1e-8) p_this(0, i) = 1e-8;
			}

			MatrixXd weightMatrix = p_this.replicate(D, 1);
			MatrixXd wS1 = weightMatrix.array() * S1.array();
			MatrixXd wS2 = weightMatrix.array() * S2.array();
			//	std::cout << "wS1 = " << wS1 << std::endl;

			MatrixXd A = wS1 * wS2.transpose();
			Eigen::JacobiSVD<MatrixXd, Eigen::NoQRPreconditioner> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
			//std::cout << "U = " << svd.matrixU() << std::endl;
			//std::cout << "V = " << svd.matrixV() << std::endl;
			MatrixXd C = MatrixXd::Identity(D, D);
			C(D - 1, D - 1) = (svd.matrixU() * svd.matrixV().transpose()).determinant();
			R = svd.matrixU() * C * svd.matrixV().transpose();
			//	std::cout << "R = " << R << std::endl;

			mu = wS1.rowwise().norm().norm() / wS2.rowwise().norm().norm();
			//	std::cout << "mu = " << mu << std::endl;
		}

		if (mu < 0.2 || mu > 5.0) continue; // we assume inliers should not have so large change

		MatrixXd distance_this(1, N);
		distance_this.block(0, 0, 1, benchPid) = distance.block(0, 0, 1, benchPid);
		distance_this(0, benchPid) = 0.0;
		distance_this.block(0, benchPid + 1, 1, N - benchPid - 1) = distance.block(0, benchPid, 1, N - benchPid - 1);
		//	std::cout << "distance_this = " << distance_this << std::endl;
		std::vector<bool> inliersMask_this;
		int N_inliersthis = 0;
		for (int i = 0; i < N; i++)
		{
			if (distance_this(0, i) < inliersThreshold)
			{
				inliersMask_this.push_back(true);
				N_inliersthis++;
			}
			else
			{
				inliersMask_this.push_back(false);
			}
		}
		if (N_inliersthis < LeastNumberOfInlierRANSACTrial) continue;

		MatrixXd t = X1.col(benchPid) / mu - R * X2.col(benchPid);
		//	std::cout << "R = " << R << std::endl;
		//	std::cout << "t = " << t << std::endl;

		double dq_this[8];
		GenerateDQfromEigenRt(R, t, dq_this, (int)D);

		double sumP = 0.0;
		for (int i = 0; i < N; i++)
			sumP += exp(-0.5 * distance_this(0, i) * distance_this(0, i) / (inliersThreshold * inliersThreshold));

		for (int i = 0; i < N; i++)
		{
			if (distance_this(0, i) < distance_all(0, i) && inliersMask_this.at(i))
			{
				for (int l = 0; l < dqsize; l++)  dq_points(i, l) = dq_this[l];
				mu_points(i, 0) = mu;
				distance_all(0, i) = distance_this(0, i);
				this->mask_R1PRNSC.at(i) = true;
				SumRNSCWeight(0, i) = sumP;
			}
		}

		int N_RNSCinliers = 0;
		for (int i = 0; i < N; i++)
			if (this->mask_R1PRNSC.at(i)) N_RNSCinliers++;

		inliersRatio = (double)N_RNSCinliers / (double)N;
		double temp = 1.0 - (double)LeastNumberOfInlierRANSACTrial / ((1.0 - inliersRatio) * (double)N + 1e-8);
		temp = max(0.01, temp);
		punish = log(1.0 - 0.95) / log(temp);
	}
	if (inliersRatio < 0.05)
	{
		std::cerr << "R1P-RANC cannot find enough number of inliers, will not perform EMDQ..." << std::endl;
		std::cerr << "inliersRatio = " << inliersRatio << std::endl;
		return 0;
	}

	// std::cout << "dq_points = " << dq_points << std::endl;
	// EMDQ
	size_t M = (N - 1 > NeighborCount) ? NeighborCount : N - 1;

	MatrixXd Kraw;
	MatrixXi neighborIds;
	GenerateKaw(X1, X2, M, Kraw, neighborIds, beta);

	/*	for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < 2 * M; j++)
			{
				int npid = neighborIds(i, j);
				if (this->mask_R1PRNSC.at(npid) == false)
					Kraw(i, j) = 0.1 * Kraw(i, j);
			}
		}
	*/
	//	std::cout << "Kraw = " << Kraw << std::endl;

	double sigma2 = inliersThreshold * inliersThreshold;
	double gamma = inliersRatio;
	gamma = (gamma > 0.95) ? 0.95 : gamma;
	gamma = (gamma < 0.05) ? 0.05 : gamma;

	MatrixXd p = (distance_all.array() * distance_all.array() / (-2.0 * sigma2)).exp();
	p = p.array() / ((p.array() + a * sigma2 * (1 - gamma) / gamma).array());
	p = p.array() * SumRNSCWeight.array();
	for (int i = 0; i < N; i++)
		if (p(0, i) < 1e-5) p(0, i) = 1e-5;
	MatrixXd distance2 = MatrixXd::Zero(1, N);
	int N_inliers = 0;
	MatrixXd p_old = p;
	MatrixXd X1_dq;
	for (int iter = 0; iter < EMMaxIterNumber; iter++)
		//	for (int iter = 0; iter < 1; iter++)
	{
		MatrixXd p_neighbor;
		GenerateNeighborMatrixFromVector(p, neighborIds, p_neighbor);
		//	std::cout << "p_neighbor = " << p_neighbor << std::endl;
		MatrixXd Kraw_p = p_neighbor.array() * Kraw.array();
		//	std::cout << "Kraw_p = " << Kraw_p << std::endl;
		MatrixXd K = Kraw_p.array() / ((Kraw_p.rowwise().sum().array() + 1e-9).replicate(1, Kraw_p.cols()));

		//E-step
	//	std::cout << "1 dq_points = " << dq_points << std::endl;
		MatrixXd dq_points_neighbor;
		for (int l = 0; l < dqsize; l++)
		{
			GenerateNeighborMatrixFromVector(dq_points.col(l), neighborIds, dq_points_neighbor);
			dq_points.col(l) = (dq_points_neighbor.array() * K.array()).rowwise().sum();
			//	std::cout << "dq_points.col(l) = " << dq_points.col(l) << std::endl;
		}
		//	std::cout << "2 dq_points = " << dq_points << std::endl;
		MatrixXd mu_points_neighbor;
		GenerateNeighborMatrixFromVector(mu_points, neighborIds, mu_points_neighbor);
		mu_points = (mu_points_neighbor.array() * K.array()).rowwise().sum();

		WarpEigenPointsFromDQ(X2, dq_points, X1_dq);
		X1_dq = X1_dq.array() * mu_points.transpose().replicate(D, 1).array();
		//	std::cout << "X1_dq = " << X1_dq << std::endl;

		MatrixXd X_diff = X1 - X1_dq;
		//	std::cout << "X_diff = " << X_diff << std::endl;

		distance2 = (X_diff.array() * X_diff.array()).colwise().sum();
		//	std::cout << "distance2 = " << distance2.array().sqrt() << std::endl;
		p = (distance2.array() / (-2.0 * sigma2)).exp();
		p = p.array() / ((p.array() + a * (2.0 * 3.1415926 * sigma2) * (1 - gamma) / gamma).array());
		for (int i = 0; i < N; i++)
			if (p(0, i) < 1e-5) p(0, i) = 1e-5;
		//	std::cout << "p = " << p << std::endl;

		sigma2 = (p.array() * distance2.array()).rowwise().sum()(0, 0) / ((p.array()).rowwise().sum()(0, 0) + 1e-8);
		//	std::cout << "(p.array() * distance2.array()).rowwise().sum() = " << (p.array() * distance2.array()).rowwise().sum() << std::endl;
		//	std::cout << "(p.array()).rowwise().sum() = " << (p.array()).rowwise().sum() << std::endl;

		N_inliers = 0;
		for (int i = 0; i < N; i++)
		{
			if (p(0, i) > pThreshold && distance2(0, i) < inliersThreshold* inliersThreshold)
				N_inliers++;
		}
		gamma = (double)N_inliers / (double)N;
		if (gamma > 0.95) gamma = 0.95;
		else if (gamma < 0.05) gamma = 0.05;

	//	std::cout << "sigma2 = " << sigma2 << ", gamma = " << gamma << std::endl;

		#pragma omp parallel for
		for (int i = 0; i < N; i++)
		{
			double xdiff[3] = { 0.0, 0.0, 0.0 };
			for (int l = 0; l < D; l++) xdiff[l] = X_diff(l, i) / mu_points(i, 0);
			double Qdiff[8];
			trans2dquat(xdiff, Qdiff, (int)D);
			double  dq_points_this_old[8], dq_points_this_new[8];
			for (int l = 0; l < dqsize; l++) dq_points_this_old[l] = dq_points(i, l);
			DQmult(dq_points_this_old, Qdiff, dq_points_this_new, (int)D);
			for (int l = 0; l < dqsize; l++) dq_points(i, l) = dq_points_this_new[l];
		}

		double pdiff = (p - p_old).array().abs().mean();
		if (iter > 0 && pdiff < pdiffthreshold)
			break;
		p_old = p;
	}
	for (int i = 0; i < N; i++)
	{
		if (p(0, i) > pThreshold&& distance2(0, i) < inliersThreshold* inliersThreshold)
			this->mask_EMDQ.at(i) = true;
		else
			this->mask_EMDQ.at(i) = false;

		std::vector<double> dq_points_this;
		dq_points_this.reserve(dqsize);
		for (int l = 0; l < dqsize; l++) dq_points_this.push_back(dq_points(i, l));
		this->dq_matches.push_back(dq_points_this);
		this->mu_matches.push_back(mu_points(i, 0));
	}

	this->p_matches.clear();
	this->p_matches.reserve(N);
	for (int i = 0; i < N; i++)
		this->p_matches.emplace_back(p(0, i));

	this->error2_matches.clear();
	this->error2_matches.reserve(N);
	for (int i = 0; i < N; i++)
		this->error2_matches.emplace_back((distance2(0, i)));

	this->numberOfEMDQInliers = N_inliers;
	return N_inliers;
}


void WarpEigenPointsFromDQ(const MatrixXd& X_in, const MatrixXd& dq_in
						, MatrixXd& X_out)
{
	int N = X_in.cols();
	int D = X_in.rows();
	X_out = MatrixXd::Zero(D, N);

	for (int i = 0; i < N; i++)
	{
		double x_in[3] = { 0.0, 0.0, 0.0 };
		for (int l = 0; l < D; l++) x_in[l] = X_in(l, i);

		double dq_this[8] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
		int dqsize = (D == 2) ? 4 : 8;
		for (int l = 0; l < dqsize; l++) dq_this[l] = dq_in(i, l);
		double x_out[3] = { 0.0, 0.0, 0.0 };
		WarpPosByDq(dq_this, x_in, x_out, (int)D);
		for (int l = 0; l < D; l++) X_out(l, i) = x_out[l];
	}
}

double distance2betweenpoints(const double* x_in, const double* y_in, const int D)
{
	double dis2 = 0.0;
	for (int l = 0; l < D; l++) dis2 += (x_in[l] - y_in[l]) * (x_in[l] - y_in[l]);
	return dis2;
}

void GenerateNeighborMatrixFromVector(const MatrixXd& p, const MatrixXi& neighborIds
						, MatrixXd& p_neighbor)
{
	bool flag_isvertical = true;
	if (p.rows() == 1) flag_isvertical = false;

	int N = neighborIds.rows();
	int M = neighborIds.cols();
	p_neighbor = MatrixXd::Zero(N, M);

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < M; j++)
		{
			int npid = neighborIds(i, j);
			if (npid >= 0 && npid < N)
			{
				p_neighbor(i, j) = (flag_isvertical) ? p(npid, 0) : p(0, npid);
			}
		}
	}
}

void GenerateWeightMatrixUsingSort(const MatrixXd& X, MatrixXd& K, MatrixXi& neighborIds, const int M, const double beta_in)
{
	int N = X.cols();
	int D = X.rows();
	K = MatrixXd::Zero(N, M);
	neighborIds = MatrixXi::Zero(N, M);

	for (int i = 0; i < N; i ++)
	{
		double coord[3] = { 0.0, 0.0, 0.0 };
		for (int l = 0; l < D; l++) coord[l] = X(l, i);
		
		vector<float> dis2;
		dis2.reserve(N);
		for (int j = 0; j < N; j++)
		{
			double coord2[3] = { 0.0, 0.0, 0.0 };
			for (int l = 0; l < D; l++) coord2[l] = X(l, j);
			double dis2_this = distance2betweenpoints(coord, coord2, D);
			dis2.emplace_back((float)dis2_this);
		}
		vector<int> ratedidx;
		ratedidx.reserve(N);
		cv::sortIdx(dis2, ratedidx, cv::SortFlags::SORT_ASCENDING);
		for (int j = 0; j < M; j++)
		{
			int npid = ratedidx.at(j);
			neighborIds(i, j) = npid;
			if (npid == i)
			{
				K(i, j) = 0.0;
			}
			else
			{
				double dis2_this = dis2.at(npid);// distance2betweenpoints(coord, coord2, D);
				K(i, j) = exp(-beta_in * dis2_this) + 1e-8;
			}
		}
	}
}


void GenerateKaw(const MatrixXd& X1, const MatrixXd& X2, const int M
				, MatrixXd& Kraw, MatrixXi& neighborIds
				, const double beta_in)
{
	int N = X1.cols();
//	int D = X1.rows();
	MatrixXd K1(N, M);
	MatrixXi neighborIds1(N, M);
//	GenerateWeightMatrixUsingKDtree(X1, K1, neighborIds1, M);
	GenerateWeightMatrixUsingSort(X1, K1, neighborIds1, M, beta_in);
	MatrixXd K2(N, M);
	MatrixXi neighborIds2(N, M);
//	GenerateWeightMatrixUsingKDtree(X2, K2, neighborIds2, M);
	GenerateWeightMatrixUsingSort(X2, K2, neighborIds2, M, beta_in);
	Kraw = MatrixXd::Zero(N, 2 * M);
	neighborIds = MatrixXi::Zero(N, 2 * M);
	Kraw.block(0, 0, N, M) = K1;
	Kraw.block(0, M, N, M) = K2;
	neighborIds.block(0, 0, N, M) = neighborIds1;
	neighborIds.block(0, M, N, M) = neighborIds2;
}



void GenerateDQfromEigenRt(const MatrixXd& R, const MatrixXd& t
	, double* dq, const int D)
{
	double* R0, *t0;
	int Rsize = 4, tsize = 2;	
	if (D == 3)
	{
		Rsize = 9;
		tsize = 3;
	}
	R0 = new double[Rsize];
	t0 = new double[tsize];
	
	for (int l1 = 0; l1 < D; l1++)
		for (int l2 = 0; l2 < D; l2++)
			R0[D * l1 + l2] = R(l1, l2);
	for (int l = 0; l < D; l ++) t0[l] = t(l);

	GenerateDQfromRt(R0, t0, dq, (int)D);

	delete[] R0;
	delete[] t0;
}

int EMDQSLAM::CEMDQ::GetNumberOfInliers()
{
	return this->numberOfEMDQInliers;
}

int EMDQSLAM::CEMDQ::GetNumberOfTotalMatches()
{
	return (int)this->matchUV1.size();
}



bool EMDQSLAM::CEMDQ::GetDeformedCoord(const double* x_in, double* x_out)
{
	if (this->matchUV2.size() == 0) return false;

	int N = this->matchUV2.size();
	int D = this->matchUV2.at(0).size();
	int dqsize = (D == 2) ? 4 : 8;

	double dq[8] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
	double mu = 0.0;
	double sumweight = 0.0;
	std::vector<double> w;
	w.reserve(N);

	for (int i = 0; i < N; i ++)
	{
		double coord[3];
		for (int l = 0; l < D; l++) coord[l] = this->matchUV2.at(i).at(l);
		double dis2 = distance2betweenpoints(x_in, coord, D);
		double weight = (this->mask_EMDQ.at(i)) ? exp(-beta2 * dis2) + 1e-10 : 1e-10;

		w.push_back(weight);
		sumweight += weight;
	}

	for (int i = 0; i < N; i++)
		w.at(i) = w.at(i) / sumweight;

	for (int i = 0; i < N; i ++)
	{
		double dq_matches_this[8] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
		for (int l = 0; l < dqsize; l++) dq_matches_this[l] = this->dq_matches.at(i).at(l);
		double mu_matches_this = this->mu_matches.at(i);

		for (int l = 0; l < dqsize; l++) dq[l] += w.at(i) * dq_matches_this[l];
		mu += w.at(i) * mu_matches_this;
	}
	NormlizeDQ(dq, (int)D);

	WarpPosByDq(dq, x_in, x_out, (int)D);
	for (int l = 0; l < D; l++) x_out[l] *= mu;
	if (D == 2) x_out[2] = 0.0;

	return true;
}

//									
bool EMDQSLAM::CEMDQ::ComputeDeformationFieldAtGivenPosition(const double* x_in, double* x_out
									, double& sigma2_change_out
									, double* dq_change_out, double& mu_change_out)
{
	if (this->matchUV2.size() == 0) return false;

	int N = this->matchUV2.size();
	int D = this->matchUV2.at(0).size();
	int dqsize = (D == 2) ? 4 : 8;

	double dq[8] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
	double mu = 0.0;
	double sumweight = 0.0;
	std::vector<double> w;
	w.reserve(N);
	double sigma2_featurebased = 1e+6;

	for (int i = 0; i < N; i++)
	{
		double coord[3];
		for (int l = 0; l < D; l++) coord[l] = this->matchUV2.at(i).at(l);
		double dis2 = distance2betweenpoints(x_in, coord, D);
		double weight = (this->mask_EMDQ.at(i)) ? exp(-beta2 * dis2) + 1e-10 : 1e-10;
		
		w.emplace_back(weight);
		sumweight += weight;

		double sigma2_thisfeature = this->error2_matches.at(i) + exp(0.003 * dis2 / (this->scale * this->scale));
		if (sigma2_featurebased > sigma2_thisfeature) sigma2_featurebased = sigma2_thisfeature;
	}

	for (int i = 0; i < N; i++)
		w.at(i) = w.at(i) / sumweight;

	for (int i = 0; i < N; i++)
	{
		double dq_matches_this[8] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};;
		for (int l = 0; l < dqsize; l++) dq_matches_this[l] = this->dq_matches.at(i).at(l);
		double mu_matches_this = this->mu_matches.at(i);

		for (int l = 0; l < dqsize; l++) dq[l] += w.at(i) * dq_matches_this[l];
		mu += w.at(i) * mu_matches_this;
	}
	NormlizeDQ(dq, (int)D);
	
	WarpPosByDq(dq, x_in, x_out, (int)D);
	for (int l = 0; l < D; l++) x_out[l] *= mu;
	
	if (D == 2) x_out[2] = 0.0;

	sigma2_change_out = sigma2_featurebased;
	
	for (int l = 0; l < dqsize; l++) dq_change_out[l] = dq[l];
	mu_change_out = mu;
	return true;
}


void EMDQSLAM::CEMDQ::VisulizeResults(const std::vector<bool> mask, const cv::Mat& inputMat1, cv::Mat& outputMat1, const cv::Mat& inputMat2, cv::Mat& outputMat2)
{
	inputMat1.copyTo(outputMat1);
	inputMat2.copyTo(outputMat2);
	for (size_t i = 0; i < mask.size(); i++)
	{
		Point2f pt1((float)this->matchUV1.at(i).at(0), (float)this->matchUV1.at(i).at(1));
		Point2f pt2((float)this->matchUV2.at(i).at(0), (float)this->matchUV2.at(i).at(1));
		if (mask.at(i))
		{
			cv::line(outputMat1, pt2, pt1, Scalar(0, 255, 255), 2);
			cv::line(outputMat2, pt1, pt2, Scalar(0, 255, 255), 2);
			cv::circle(outputMat1, pt1, 3, Scalar(255, 0, 0), 3);
			cv::circle(outputMat2, pt2, 3, Scalar(255, 0, 0), 3);
		}
		else
		{
			cv::line(outputMat1, pt2, pt1, Scalar(0, 0, 0), 2);
			cv::line(outputMat2, pt1, pt2, Scalar(0, 0, 0), 2);
			cv::circle(outputMat1, pt1, 3, Scalar(0, 0, 0), 3);
			cv::circle(outputMat2, pt2, 3, Scalar(0, 0, 0), 3);
		}
	}
}

void EMDQSLAM::CEMDQ::Visulize2DDeformationField(const cv::Mat& inputMat2, cv::Mat& outputMat2
										, const int step)
{
	inputMat2.copyTo(outputMat2);
	int H = inputMat2.rows;
	int W = inputMat2.cols;
	for (int y = 0; y < H; y += step)
		for (int x = 0; x < W; x += step)
	{
		double coord1[2] = {(double)x, (double)y};
		double coord2[2] = {0.0, 0.0};
		double sigma2_change = 1.0;
		double dq_change[4], mu_change;
		
		if (1) // only draw coordinates
		{
			this->GetDeformedCoord(coord1, coord2);
			Point2f pt1((float)coord1[0], (float)coord1[1]);
			Point2f pt2((float)coord2[0], (float)coord2[1]);
			cv::arrowedLine(outputMat2, pt1, pt2, Scalar(0, 255, 255), 2);
		}
		else // draw coordinates and uncertainties
		{
			this->ComputeDeformationFieldAtGivenPosition(coord1, coord2
					, sigma2_change, dq_change, mu_change);
		
			double sigma_this = sqrt(sigma2_change);
			double u = 255.0 - 10.0 * sigma_this;
			if (u < 0.0) u = 0.0;
			if (u > 255.0) u = 255.0;
			unsigned char c = (unsigned char)u;
		
			Point2f pt1((float)coord1[0], (float)coord1[1]);
			Point2f pt2((float)coord2[0], (float)coord2[1]);
			cv::arrowedLine(outputMat2, pt1, pt2, Scalar(c, 0, 255 - c), 2);
		}
	}
}



// temp for comparsion

bool EMDQSLAM::CEMDQ::GetDisplacedCoord(const double* x_in, double* x_out)
{
	if (this->matchUV2.size() == 0) return false;

	int N = this->matchUV2.size();
	int D = this->matchUV2.at(0).size();

	double displacement[3] = { 0.0, 0.0, 0.0 };
	double sumweight = 0.0;
	std::vector<double> w;
	w.reserve(N);

	for (int i = 0; i < N; i ++)
	{
		double coord[3];
		for (int l = 0; l < D; l++) coord[l] = this->matchUV2.at(i).at(l);
		double dis2 = distance2betweenpoints(x_in, coord, D);
		double weight = (this->mask_EMDQ.at(i)) ? exp(-beta2 * dis2) + 1e-10 : 1e-10;

		w.push_back(weight);
		sumweight += weight;
	}

	for (int i = 0; i < N; i++)
		w.at(i) = w.at(i) / sumweight;

	for (int i = 0; i < N; i ++)
	{
		for (int l = 0; l < D; l++) displacement[l] += w.at(i) * (this->matchUV1.at(i).at(l) - this->matchUV2.at(i).at(l));
	}
	
	for (int l = 0; l < D; l++) x_out[l] = x_in[l] + displacement[l];
	if (D == 2) x_out[2] = 0.0; 
	return true;
}

void EMDQSLAM::CEMDQ::Visulize2DDisplacementInterplation(const cv::Mat& inputMat2, cv::Mat& outputMat2, const int step)
{
	inputMat2.copyTo(outputMat2);
	int H = inputMat2.rows;
	int W = inputMat2.cols;
	for (int y = 0; y < H; y += step)
		for (int x = 0; x < W; x += step)
	{
		double coord1[2] = {(double)x, (double)y};
		double coord2[2] = {0.0, 0.0};
		
		this->GetDisplacedCoord(coord1, coord2);
		Point2f pt1((float)coord1[0], (float)coord1[1]);
		Point2f pt2((float)coord2[0], (float)coord2[1]);
		cv::arrowedLine(outputMat2, pt1, pt2, Scalar(0, 255, 255), 2);
	}
}

