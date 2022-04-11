/************************************************************************/
// simplified version of dual quaternion-related computations
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

#include "DualQ.h"
#include <iostream>
using namespace std;

#include <Eigen/Dense>
using namespace Eigen;
using Eigen::MatrixXd;
using Eigen::MatrixXi;

void MatrixTranspose(const double* dcm, double* dcm_transpose, const int D)
{
	for (int l1 = 0; l1 < D; l1 ++)
		for (int l2 = 0; l2 < D; l2++)
			dcm_transpose[D * l1 + l2] = dcm[D * l2 + l1];
}

void MultiplyMatrix(const double* dcm1_in, const double* dcm2_in, double* dcm_out, const int D)
{
	if (D == 3)
	{
		dcm_out[0] = dcm1_in[0] * dcm2_in[0] + dcm1_in[1] * dcm2_in[3] + dcm1_in[2] * dcm2_in[6];
		dcm_out[1] = dcm1_in[0] * dcm2_in[1] + dcm1_in[1] * dcm2_in[4] + dcm1_in[2] * dcm2_in[7];
		dcm_out[2] = dcm1_in[0] * dcm2_in[2] + dcm1_in[1] * dcm2_in[5] + dcm1_in[2] * dcm2_in[8];
		dcm_out[3] = dcm1_in[3] * dcm2_in[0] + dcm1_in[4] * dcm2_in[3] + dcm1_in[5] * dcm2_in[6];
		dcm_out[4] = dcm1_in[3] * dcm2_in[1] + dcm1_in[4] * dcm2_in[4] + dcm1_in[5] * dcm2_in[7];
		dcm_out[5] = dcm1_in[3] * dcm2_in[2] + dcm1_in[4] * dcm2_in[5] + dcm1_in[5] * dcm2_in[8];
		dcm_out[6] = dcm1_in[6] * dcm2_in[0] + dcm1_in[7] * dcm2_in[3] + dcm1_in[8] * dcm2_in[6];
		dcm_out[7] = dcm1_in[6] * dcm2_in[1] + dcm1_in[7] * dcm2_in[4] + dcm1_in[8] * dcm2_in[7];
		dcm_out[8] = dcm1_in[6] * dcm2_in[2] + dcm1_in[7] * dcm2_in[5] + dcm1_in[8] * dcm2_in[8];
	}
	else if (D == 2)
	{
		dcm_out[0] = dcm1_in[0] * dcm2_in[0] + dcm1_in[1] * dcm2_in[2];
		dcm_out[1] = dcm1_in[0] * dcm2_in[1] + dcm1_in[1] * dcm2_in[3];
		dcm_out[2] = dcm1_in[2] * dcm2_in[0] + dcm1_in[3] * dcm2_in[2];
		dcm_out[3] = dcm1_in[2] * dcm2_in[1] + dcm1_in[3] * dcm2_in[3];
	}
}

void MultiplyMatrixAndVector(const double* dcm_in, const double* t_in
	, double* dcmt_out, const int D)
{
	if (D == 2)
	{
		dcmt_out[0] = dcm_in[0] * t_in[0] + dcm_in[1] * t_in[1];
		dcmt_out[1] = dcm_in[2] * t_in[0] + dcm_in[3] * t_in[1];
	}
	else if (D == 3)
	{
		dcmt_out[0] = dcm_in[0] * t_in[0] + dcm_in[1] * t_in[1] + dcm_in[2] * t_in[2];
		dcmt_out[1] = dcm_in[3] * t_in[0] + dcm_in[4] * t_in[1] + dcm_in[5] * t_in[2];
		dcmt_out[2] = dcm_in[6] * t_in[0] + dcm_in[7] * t_in[1] + dcm_in[8] * t_in[2];
	}
}

void MultiplyVectorAndMatrix(const double* t_in, const double* dcm_in
				, double* tdcm_out, const int D)
{
	if (D == 2)
	{
		tdcm_out[0] = dcm_in[0] * t_in[0] + dcm_in[2] * t_in[1];
		tdcm_out[1] = dcm_in[1] * t_in[0] + dcm_in[3] * t_in[1];
	}
	else if (D == 3)
	{
		tdcm_out[0] = dcm_in[0] * t_in[0] + dcm_in[3] * t_in[1] + dcm_in[6] * t_in[2];
		tdcm_out[1] = dcm_in[1] * t_in[0] + dcm_in[4] * t_in[1] + dcm_in[7] * t_in[2];
		tdcm_out[2] = dcm_in[2] * t_in[0] + dcm_in[5] * t_in[1] + dcm_in[8] * t_in[2];
	}
}

void MultiplyVectorAndMatrixTranspose(const double* t_in, const double* dcm_in, double* tdcm_transpose_out, const int D)
{
	if (D == 2)
	{
		tdcm_transpose_out[0] = dcm_in[0] * t_in[0] + dcm_in[1] * t_in[1];
		tdcm_transpose_out[1] = dcm_in[2] * t_in[0] + dcm_in[3] * t_in[1];
	}
	else if (D == 3)
	{
		tdcm_transpose_out[0] = dcm_in[0] * t_in[0] + dcm_in[1] * t_in[1] + dcm_in[2] * t_in[2];
		tdcm_transpose_out[1] = dcm_in[3] * t_in[0] + dcm_in[4] * t_in[1] + dcm_in[5] * t_in[2];
		tdcm_transpose_out[2] = dcm_in[6] * t_in[0] + dcm_in[7] * t_in[1] + dcm_in[8] * t_in[2];
	}
}


void Qmult(const double* q, const double* r, double* s_out)
{
	s_out[0] = q[0] * r[0] - q[1] * r[1] - q[2] * r[2] - q[3] * r[3];
	s_out[1] = r[0] * q[1] + q[0] * r[1] - (q[2] * r[3] - q[3] * r[2]);
	s_out[2] = r[0] * q[2] + q[0] * r[2] - (q[3] * r[1] - q[1] * r[3]);
	s_out[3] = r[0] * q[3] + q[0] * r[3] - (q[1] * r[2] - q[2] * r[1]);
}

void Qmult2d_type1(const double* q, const double* r, double* s_out)
{
	s_out[0] = q[0] * r[0] - q[1] * r[1];
	s_out[1] = q[1] * r[0] + q[0] * r[1];
}

void Qmult2d_type2(const double* q, const double* r, double* s_out)
{
	s_out[0] = q[0] * r[0] + q[1] * r[1]; 
	s_out[1] = q[0] * r[1] - q[1] * r[0];
}

void Qmult2d_type3(const double* q, const double* r, double* s_out)
{
	s_out[0] = q[0] * r[0] - q[1] * r[1];
	s_out[1] = q[1] * r[0] + q[0] * r[1];
}

void DQmult(const double* dq, const double* dr, double* ds, const int D)
{
	if (D == 3)
	{
		Qmult(dq, dr, ds);
		double ds21[4], ds22[4];
		Qmult(dq, dr + 4, ds21);
		Qmult(dq + 4, dr, ds22);
		for (int l = 0; l < 4; l++) ds[l + 4] = ds21[l] + ds22[l];
	}
	else if (D == 2)
	{
		Qmult2d_type1(dq, dr, ds); 
		double ds21[2], ds22[2]; 
		Qmult2d_type2(dq, dr + 2, ds21); 
		Qmult2d_type3(dq + 2, dr, ds22); 
		for (int l = 0; l < 2; l++) ds[l + 2] = ds21[l] + ds22[l]; 
	}
} 

void DQconj(const double* dq_in, double* dq_out, const int D)
{
	if (D == 3)
	{
		dq_out[0] = dq_in[0];
		for (int l = 1; l < 5; l++) dq_out[l] = -dq_in[l];
		for (int l = 5; l < 8; l++) dq_out[l] = dq_in[l];
	}
	else if (D == 2)
	{
		dq_out[0] = dq_in[0];
		dq_out[1] = -dq_in[1];
		dq_out[2] = dq_in[2];
		dq_out[3] = dq_in[3];
	}
}

void pos2dquat(const double* pos_in, double* dq_out, const int D)
{
	if (D == 3)
	{
		dq_out[0] = 1.0;
		for (int l = 1; l < 5; l++)	dq_out[l] = 0.0;
		for (int l = 5; l < 8; l++)	dq_out[l] = pos_in[l - 5];
	}
	else if (D == 2)
	{
		dq_out[0] = 1.0;
		dq_out[1] = 0.0;
		for (int l = 2; l < 4; l++)	dq_out[l] = pos_in[l - 2];
	}
}

void NormlizeDQ(double* dq, const int D)
{ 
	if (D == 3)
	{
		double s = dq[0] * dq[0] + dq[1] * dq[1] + dq[2] * dq[2] + dq[3] * dq[3]; 
		s = sqrt(s); 
		s = 1.0 / s; 
		for (int l = 0; l < 8; l++)	dq[l] = s * dq[l]; 
	}
	else if (D == 2)
	{
		double s = dq[0] * dq[0] + dq[1] * dq[1]; 
		s = sqrt(s); 
		s = 1.0 / s; 
		for (int l = 0; l < 4; l++)	dq[l] = s * dq[l]; 
	}
}

void WarpPosByDq(const double* dq_in, const double* pos_in, double* pos_out, const int D)
{
	if (D == 3)
	{
		double dq_conj[8];
        DQconj(dq_in, dq_conj, D);
		double Qpos[8];
		pos2dquat(pos_in, Qpos, D);
		double Q1[8], Q2[8];
		DQmult(dq_conj, Qpos, Q1, D);
		DQmult(Q1, dq_in, Q2, D);
		NormlizeDQ(Q2, D);
		for (int l = 0; l < 3; l ++) pos_out[l] = Q2[l + 5];
	}
	else if (D == 2)
	{
		double dq_conj[4];
        DQconj(dq_in, dq_conj, D);
		double Qpos[4];
		pos2dquat(pos_in, Qpos, D);
		double Q1[4], Q2[4];
		DQmult(dq_conj, Qpos, Q1, D);
		DQmult(Q1, dq_in, Q2, D);
		NormlizeDQ(Q2, D);
		for (int l = 0; l < 2; l ++) pos_out[l] = Q2[l + 2];
	}
}

void WarpPosByDqmu(const double* dq_in, const double mu_in, const double* pos_in, double* pos_out, const int D)
{
	WarpPosByDq(dq_in, pos_in, pos_out, D);
	for (int l = 0; l < D; l++) pos_out[l] *= mu_in;
}

void WarpBackPosByDqmu(const double* dq_in, const double mu_in, const double* pos_in, double* pos_out, const int D)
{
	int dqsize = (D == 2) ? 4 : 8;
	double* dqinv = new double[dqsize];
	DQinv(dq_in, dqinv, D);
	double* coord = new double[D];
	for (int l = 0; l < D; l++) coord[l] = pos_in[l] / mu_in;
	WarpPosByDq(dqinv, coord, pos_out, D);
	delete[] dqinv;
	delete[] coord;
}



void DQinv(const double* dq_in, double* dq_inv_out, const int D)
{
	if (D == 3)
	{
		dq_inv_out[0] = dq_in[0];
		for (int l = 1; l < 4; l ++) dq_inv_out[l] = -dq_in[l];
		dq_inv_out[4] = dq_in[4];
		for (int l = 5; l < 8; l ++) dq_inv_out[l] = -dq_in[l];
	}
	else if (D == 2)
	{
		dq_inv_out[0] = dq_in[0];
		for (int l = 1; l < 4; l++) dq_inv_out[l] = -dq_in[l];
	}
}

void rotMatrix2dquat(const double* R_in, double* dq_out, const int D)
{
	if (D == 3)
	{
		dq_out[0] = 0.5 * sqrt(1 + R_in[0] + R_in[4] + R_in[8]);
		double temp = 0.25 / dq_out[0];
		dq_out[1] = temp * (R_in[7] - R_in[5]);
		dq_out[2] = temp * (R_in[2] - R_in[6]);
		dq_out[3] = temp * (R_in[3] - R_in[1]);
		for (int l = 4; l < 8; l ++) dq_out[l] = 0.0;
	}
	else if (D == 2)
	{
		dq_out[0] = 0.5 * sqrt(2.0 + R_in[0] + R_in[3]);
		double temp = 0.25 / dq_out[0];
		dq_out[1] = temp * (R_in[2] - R_in[1]);
		for (int l = 2; l < 4; l ++) dq_out[l] = 0.0;
	}
}

void dquat2rotMatrix(const double* dq_in, double* R_out, const int D)
{
	if (D == 3)
	{
		R_out[0] = 1.0 - 2.0 * dq_in[2] * dq_in[2] - 2.0 * dq_in[3] * dq_in[3]; 
		R_out[1] = 2.0 * dq_in[1] * dq_in[2] - 2.0 * dq_in[3] * dq_in[0]; 
		R_out[2] = 2.0 * dq_in[1] * dq_in[3] + 2.0 * dq_in[2] * dq_in[0]; 
		R_out[3] = 2.0 * dq_in[1] * dq_in[2] + 2.0 * dq_in[3] * dq_in[0]; 
		R_out[4] = 1.0 - 2.0 * dq_in[1] * dq_in[1] - 2.0 * dq_in[3] * dq_in[3]; 
		R_out[5] = 2.0 * dq_in[2] * dq_in[3] - 2.0 * dq_in[1] * dq_in[0]; 
		R_out[6] = 2.0 * dq_in[1] * dq_in[3] - 2.0 * dq_in[2] * dq_in[0];
		R_out[7] = 2.0 * dq_in[2] * dq_in[3] + 2.0 * dq_in[1] * dq_in[0]; 
		R_out[8] = 1.0 - 2.0 * dq_in[1] * dq_in[1] - 2.0 * dq_in[2] * dq_in[2]; 
	}
	else if (D == 2)
	{
		R_out[0] = 1.0 - 2.0 * dq_in[1] * dq_in[1]; 
		R_out[1] = -2.0 * dq_in[1] * dq_in[0]; 
		R_out[2] = -R_out[1]; 
		R_out[3] = R_out[0]; 
	}
}

void trans2dquat(const double* t_in, double* dq_out, const int D)
{
	if (D == 3)
	{
		dq_out[0] = 1.0; 
		for (int l = 1; l < 5; l++)	dq_out[l] = 0.0; 
		for (int l = 5; l < 8; l++)	dq_out[l] = 0.5 * t_in[l - 5];
	}
	else if (D == 2)
	{
		dq_out[0] = 1.0; 
		dq_out[1] = 0.0; 
		for (int l = 2; l < 4; l++)	dq_out[l] = 0.5 * t_in[l - 2];
	}
}

void dquat2trans(const double* dq_in, double* t_out, const int D)
{
	if (D == 3)
	{
		for (int l = 0; l < 3; l ++) t_out[l] = 2.0 * dq_in[l + 5];
	}
	else if (D == 2)
	{
		for (int l = 0; l < 2; l ++) t_out[l] = 2.0 * dq_in[l + 2];
	}
}

void angle2rotMatrix(const double* angle, double* R, const int D)
{
	if (D == 3)
	{
		double sx = sin(angle[2]);
		double sy = sin(angle[0]);
		double sz = sin(angle[1]);
		double cx = cos(angle[2]);
		double cy = cos(angle[0]);
		double cz = cos(angle[1]);
		R[0] = cy*cx;
		R[1] = cy*sx;
		R[2] = -sy;
		R[3] = sz*sy*cx - cz*sx;
		R[4] = sz*sy*sx + cz*cx;
		R[5] = sz*cy;
		R[6] = cz*sy*cx + sz*sx;
		R[7] = cz*sy*sx - sz*cx;
		R[8] = cz*cy;
	}
	else if (D == 2)
	{
		R[0] = cos(angle[0]);
		R[1] = sin(angle[0]);
		R[2] = -R[1];
		R[3] = R[0];
	}
}

void rotMatrix2angle(const double* R, double* angle, const int D)
{
	if (D == 3)
	{
		angle[0] = asin(-R[2]);
		angle[1] = atan2(R[5], R[8]);
		angle[2] = atan2(R[1], R[0]);
	}
	else if (D == 2)
	{
		angle[0] = atan2(R[1],R[0]);
	}
}

void angle2dquat(const double* angle, double* DQ, const int D)
{ 
	if (D == 3)
	{
		double t0 = cos(angle[2] * 0.5);
		double t1 = sin(angle[2] * 0.5);
		double t2 = cos(angle[1] * 0.5);
		double t3 = sin(angle[1] * 0.5);
		double t4 = cos(angle[0] * 0.5);
		double t5 = sin(angle[0] * 0.5);
		DQ[0] = t0 * t2 * t4 + t1 * t3 * t5;
		DQ[1] = -t0 * t3 * t4 + t1 * t2 * t5;
		DQ[2] = -t0 * t2 * t5 - t1 * t3 * t4;
		DQ[3] = -t1 * t2 * t4 + t0 * t3 * t5;
		for (int l = 4; l < 8; l++) DQ[l] = 0.0;
	}
	else if (D == 2)
	{
		DQ[0] = cos(angle[0] * 0.5);
		DQ[1] = -sin(angle[0] * 0.5);
		for (int l = 2; l < 4; l++) DQ[l] = 0.0;
	}
}

void GenerateDQfromRt(const double* R_in, const double* t_in, double* dq_out, const int D)
{
	double QR[8];
	rotMatrix2dquat(R_in, QR, D);
	double t2[3];
	MultiplyVectorAndMatrix(t_in, R_in, t2, D);
	double QT[8];
	trans2dquat(t2, QT, D);
	DQmult(QT,QR, dq_out, D);
}

void GenerateRtfromDQ(const double* dq_in, double* R_out, double* t_out, const int D)
{
	if (D == 3)
	{
		dquat2rotMatrix(dq_in, R_out, D);
		double QR[8], QR_inv[8];
		for (int l = 0; l < 4; l++) QR[l] = dq_in[l];
		for (int l = 4; l < 8; l++) QR[l] = 0.0; 
		DQinv(QR, QR_inv, D); 
		double QT2[8], t2[3]; 
		DQmult(dq_in, QR_inv, QT2, D);
		dquat2trans(QT2, t2, D); 
		MultiplyVectorAndMatrixTranspose(t2, R_out, t_out, D); 
	}
	else if (D == 2)
	{
		dquat2rotMatrix(dq_in,R_out,D);
		double QR[4], QR_inv[4];
		for (int l = 0; l < 2; l ++) QR[l] = dq_in[l];
		for (int l = 2; l < 4; l ++) QR[l] = 0.0;
		DQinv(QR, QR_inv, D);
		double QT2[4], t2[2];
		DQmult(dq_in, QR_inv, QT2, D);
		dquat2trans(QT2, t2, D);
		MultiplyVectorAndMatrixTranspose(t2, R_out, t_out, D);
	}
}

void GenerateDQfromanglet(const double* angle_in, const double* t_in, double* dq_out, const int D)
{
	int Rsize = (D == 2) ? 4 : 9;
	double* R = new double[Rsize];
	angle2rotMatrix(angle_in, R, D);
	GenerateDQfromRt(R, t_in, dq_out, D);
	delete[] R;
}

void GenerateangletfromDQ(const double* dq_in, double* angle_out, double* t_out, const int D)
{
	int Rsize = (D == 2) ? 4 : 9;
	double* R = new double[Rsize];
	GenerateRtfromDQ(dq_in, R, t_out, D);
	rotMatrix2angle(R, angle_out, D);
	delete[] R;
}

// mu_change_in * dq_change_in( mu_old_in * dq_old_in() ) = mu_new_out * dq_new_out()
void UpdateDQMuFromDQMuChange(const double* dq_old_in, const double mu_old_in
	, const double* dq_change_in, const double mu_change_in
	, double* dq_new_out, double& mu_new_out
	, const int D)
{
	int dq_size = (D == 2) ? 4 : 8;
	int R_size = (D == 2) ? 4 : 9;
	double* Rchange = new double[R_size];
	double* tchange = new double[D];
	GenerateRtfromDQ(dq_change_in, Rchange, tchange, D);

	double* temp = new double[D];
	for (int l = 0; l < D; l++) temp[l] = (1.0 - mu_old_in) / mu_old_in * tchange[l];
	double* dq_t_temp = new double[dq_size];
	trans2dquat(temp, dq_t_temp, D);
	double* dq_change_temp = new double[dq_size];
	DQmult(dq_change_in, dq_t_temp, dq_change_temp, D);
	DQmult(dq_old_in, dq_change_temp, dq_new_out, D);

	mu_new_out = mu_old_in * mu_change_in;

	delete[] Rchange;
	delete[] tchange;
	delete[] temp;
	delete[] dq_change_temp;
	delete[] dq_t_temp;
}

// mu_change_out * dq_change_out(mu_old_in * dq_old_in() ) = mu_new_in * dq_new_in()
void GetDQMuChangeFromTwoDQMu(const double* dq_new_in, const double mu_new_in
	, const double* dq_old_in, const double mu_old_in
	, double* dq_change_out, double& mu_change_out
	, const int D)
{
//	int dq_size = (D == 2) ? 4 : 8;
	int R_size = (D == 2) ? 4 : 9;
	double* Rold = new double[R_size];
	double* told = new double[D];
	double* Rnew = new double[R_size];
	double* tnew = new double[D];
	double* Rold_transpose = new double[R_size];
	double* Rchange = new double[R_size];
	double* tchange = new double[D];
	double* Rchange_told = new double[D];

	mu_change_out = mu_new_in / mu_old_in;

	GenerateRtfromDQ(dq_old_in, Rold, told, D);
	GenerateRtfromDQ(dq_new_in, Rnew, tnew, D);

	MatrixTranspose(Rold, Rold_transpose, D);
	MultiplyMatrix(Rnew, Rold_transpose, Rchange, D);
	MultiplyMatrixAndVector(Rchange, told, Rchange_told, D);

	for (int l = 0; l < D; l++)
	{
		tchange[l] = mu_old_in * (tnew[l] - Rchange_told[l]);
	}
	
	GenerateDQfromRt(Rchange, tchange, dq_change_out, D);

	delete[] Rchange_told;
	delete[] Rchange;
	delete[] tchange;
	delete[] Rold_transpose;
	delete[] Rold;
	delete[] told;
	delete[] Rnew;
	delete[] tnew;
}



