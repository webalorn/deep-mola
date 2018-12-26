#ifndef TYPES_HPP_
#define TYPES_HPP_

#include <vector>

using uint = unsigned int;
using real = double;
using Vect = std::vector<real>;
using Matrix = std::vector<Vect>;
using DataSet = std::vector<std::pair<Vect, Vect>>;

Matrix newMatrix(int, int, real defaultVal=0);

Vect vectApply(const Vect&, real (*)(real));

// These functions never check if sizes are compatible,
// it might cause unexpexted errors when operations are not mathematically allowed

real dot(const Vect&, const Vect&);

Vect operator * (const Vect&, const Vect&);
Vect operator * (const Matrix&, const Vect&);

Vect operator + (const Vect&, const Vect&);
Vect operator - (const Vect&, const Vect&);
Matrix operator + (const Matrix&, const Matrix&);

Vect operator * (Vect, real);
Matrix operator * (Matrix, real);

Matrix transpose(const Matrix&);

#endif  /* !TYPES_HPP_ */