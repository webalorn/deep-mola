#include "../headers/types.hpp"
#include <iostream>

Matrix newMatrix(int height, int width, real defaultVal) {
	return Matrix(height, Vect(width, defaultVal));
}

Vect vectApply(const Vect& inputVect, real (*func)(real)) {
	Vect outputVect;
	outputVect.reserve(inputVect.size());

	for (real funcParam : inputVect) {
		outputVect.push_back(func(funcParam));
	}
	return outputVect;
}

real dot(const Vect& v1, const Vect& v2) {
	real scalProduct = 0;
	for (real component : v1*v2) {
		scalProduct += component;
	}
	return scalProduct;
}

Vect operator * (const Vect& v1, const Vect& v2) { // v1.size() must be equal to v2.size()
	Vect outVect(v1.size());
	for (uint i = 0; i < v1.size(); i++) {
		outVect[i] = v1[i] * v2[i];
	}
	return outVect;
}
Vect operator * (const Matrix& m, const Vect& v) {
	Vect outVect(m.size());
	for (uint i = 0; i < outVect.size(); i++) {
		outVect[i] = dot(m[i], v);
	}
	return outVect;
}

Vect operator + (const Vect& v1, const Vect& v2) {
	if (v1.size() != v2.size()) {
		std::cerr << "ERROR VECT SIZES\n";
		std::cerr << "continue\n";
	}
	Vect vectOut(v1.size());
	for (uint i = 0; i < v1.size(); i++) {
		vectOut[i] = v1[i] + v2[i];
	}
	return vectOut;
}
Vect operator - (const Vect& v1, const Vect& v2) {
	return v1 + (v2 * (-1));
}
Matrix operator + (const Matrix& m1, const Matrix& m2) {
	if (m1.size() != m2.size()) {
		std::cerr << "ERROR MATRIX SIZES\n";
		std::cerr << "continue\n";
	}
	Matrix mOut(m1.size());
	for (uint iRow = 0; iRow < m1.size(); iRow++) {
		mOut[iRow] = m1[iRow] + m2[iRow];
	}
	return mOut;
}

Vect operator * (Vect v, real coeff) {
	for (real& v : v) {
		v *= coeff;
	}
	return v;
}
Matrix operator * (Matrix m, real coeff) {
	for (Vect& v : m) {
		v = v * coeff;
	}
	return m;
}

Matrix transpose(const Matrix& m) {
	Matrix tranposition = newMatrix(m[0].size(), m.size());
	for (uint i = 0; i < m.size(); i++) {
		for (uint j = 0; j < m[i].size(); j++) {
			tranposition[j][i] = m[i][j];
		}
	}
	return tranposition;
}