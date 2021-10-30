#pragma once

#include <vector>
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <unsupported/Eigen/MatrixFunctions>

using std::vector;
using Eigen::MatrixXf;

class MultiwayMatcher{

public:
	MultiwayMatcher();
	~MultiwayMatcher();
	void initialize(Eigen::MatrixXf A, vector<unsigned> numSmp);
	void set_verbose(bool verbose);

	void estimate_universe_size();
	void CLEAR();
	void get_X(Eigen::MatrixXf& X);
	void get_Y(Eigen::MatrixXf& Y);
	void get_assignments(vector<int>& assignments);
	void get_fused_counts(vector<unsigned>& fused_counts);
	void save_data();

private:
	void construct_D();
	void construct_L();
	void construct_Lnrm();

	vector<unsigned> numSmp_;
	vector<unsigned> cumSum_;

	Eigen::SparseMatrix<float> A_sp;
	Eigen::SparseMatrix<float> D_sp;
	Eigen::SparseMatrix<float> L_sp;
	Eigen::SparseMatrix<float> Lnrm_sp;


	// MatrixXf A_;
	// MatrixXf D_;
	// MatrixXf L_;
	MatrixXf Lnrm_; // Normalized Laplacian
	MatrixXf sl_; // vector of singular values of Lnrm_ (decreasing order)
	MatrixXf Vl_; // right singular vectors of Lnrm_
	MatrixXf N_; // embedding matrix (CLEAR)
	MatrixXf C_; // cluster centers (CLEAR)
	MatrixXf Y_; // lifting permutation (CLEAR)
	MatrixXf X_; // optimized pairwise permutations (CLEAR)

	unsigned m_; // estimated size of the universe
	vector<int> assignments_; // assignment of each local observation to the universe 
	vector<unsigned> fused_counts_; // observation count of each global object

	// threshold for estimating universe size
	double thresh_ = 0.50;

	bool verbose_ = true;


};