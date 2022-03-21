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

#ifndef DUALQ_H
#define DUALQ_H


void MatrixTranspose(const double* dcm, double* dcm_transpose, const int D);
void MultiplyMatrix(const double* dcm1_in, const double* dcm2_in, double* dcm_out, const int D);
void MultiplyMatrixAndVector(const double* dcm_in, const double* t_in
	, double* dcmt_out, const int D);
void MultiplyVectorAndMatrix(const double* t_in, const double* dcm_in
	, double* tdcm_out, const int D);

void MultiplyVectorAndMatrixTranspose(const double* t_in, const double* dcm_in, double* tdcm_transpose_out, const int D);


void Qmult(const double*, const double*, double*);
void Qmult2d_type1(const double* , const double*, double*);
void Qmult2d_type2(const double* , const double*, double*);
void Qmult2d_type3(const double*, const double*, double*);
void DQmult(const double* dq, const double* dr, double* ds, const int D);
void DQconj(const double* dq_in, double* dq_out, const int D);
void NormlizeDQ(double* dq, const int D);
void DQinv(const double* dq_in, double* dq_inv_out, const int D);


void pos2dquat(const double* pos_in, double* dq_out, const int D);
void rotMatrix2dquat(const double* R_in, double* dq_out, const int D);
void dquat2rotMatrix(const double* dq_in, double* R_out, const int D);
void trans2dquat(const double* t_in, double* dq_out, const int D);
void dquat2trans(const double* dq_in, double* t_out, const int D);
void angle2rotMatrix(const double* angle, double* R, const int D);
void rotMatrix2angle(const double* R, double* angle, const int D);
void angle2dquat(const double* angle, double* DQ, const int D);


void WarpPosByDq(const double* dq_in, const double* pos_in, double* pos_out, const int D);
void WarpPosByDqmu(const double* dq_in, const double mu_in, const double* pos_in, double* pos_out, const int D);
void GenerateDQfromRt(const double* R_in, const double* t_in, double* dq_out, const int D);
void GenerateRtfromDQ(const double* dq_in, double* R_out, double* t_out, const int D);
void GenerateDQfromanglet(const double* angle_in, const double* t_in, double* dq_out, const int D);
void GenerateangletfromDQ(const double* dq_in, double* angle_out, double* t_out, const int D);

void WarpBackPosByDqmu(const double* dq_in, const double mu_in, const double* pos_in, double* pos_out, const int D);

// mu_change_in * dq_change_in( mu_old_in * dq_old_in() ) = mu_new_out * dq_new_out()
void UpdateDQMuFromDQMuChange(const double* dq_old_in, const double mu_old_in
	, const double* dq_change_in, const double mu_change_in
	, double* dq_new_out, double& mu_new_out
	, const int D);

// mu_change_out * dq_change_out(mu_old_in * dq_old_in() ) = mu_new_in * dq_new_in()
void GetDQMuChangeFromTwoDQMu(const double* dq_new_in, const double mu_new_in
	, const double* dq_old_in, const double mu_old_in
	, double* dq_change_out, double& mu_change_out
	, const int D);

#endif
