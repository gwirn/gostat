package main

import (
	"fmt"
	"log"
)

/*
Dot calculates the dot product of two vectors

	:parameter
		*	a, b: vectors for which the dot product should be calculated
	:return
		*	result: dot product of the vectors
*/
func Dot(a, b []float64) float64 {
	result := 0.0
	for i := 0; i < len(a); i++ {
		result += a[i] * b[i]
	}
	return result
}

/*
GradientDescent updates the model parameters using the gradient descent algorithm

	:parameter
		*	X: data with the same format as training data
		*	y: target values
		*	theta: estimated coefficients for linear regression
		*	alpha: learning rate
	:return
		*	updatedTheta: updated coefficients based on gradient descent
*/
func GradientDescent(X [][]float64, y []float64, theta []float64, alpha float64) []float64 {
	n := len(X)
	m := len(X[0])
	updatedTheta := make([]float64, m)
	for j := 0; j < m; j++ {
		gradient := 0.0
		for i := 0; i < n; i++ {
			hypothesis := Dot(X[i], theta)
			gradient += (hypothesis - y[i]) * X[i][j]
		}
		updatedTheta[j] = theta[j] - (alpha / float64(n) * gradient)
	}
	return updatedTheta
}

/*
Predict data based on linear approximation

	:parameter
		*	X: data with the same format as training data
		*	theta: estimated coefficients for linear regression
	:return
		*	results: predicted values
*/
func predict(X [][]float64, theta []float64) []float64 {
	results := make([]float64, len(X))
	for ci, i := range X {
		for cj, j := range theta {
			results[ci] += i[cj] * j
		}
	}
	return results
}

/*
Fit linear model

	:parameter
		*	X: data with the same format as training data
		*	y: target values
		*	alpha: learning rate (0.001)
		*	nIter:	number of iterations for gradient decent (100000)
	:return:
		*	theta: estimated coefficients for linear regression
*/
func fit(X [][]float64, y []float64, alpha float64, nIter int) []float64 {
	if xSize, ySize := len(X), len(y); xSize != ySize {
		log.Fatal(fmt.Printf("Sizes of X [%d] and y [%d] don't match'", xSize, ySize))
	}
	// Initialize model theats
	predSize := len(X[0])
	theta := make([]float64, predSize)
	for i := 0; i < predSize; i++ {
		theta[i] = 0
	}

	// Train the model using gradient descent
	for i := 0; i < nIter; i++ {
		theta = GradientDescent(X, y, theta, alpha)
	}
	return theta

}

func main() {
	X := [][]float64{{1, 4}, {1, 3}, {1, 2}, {1, 1}, {1, 3}}
	y := []float64{3, 4, 5, 6, 7}
	theta := fit(X, y, 0.001, 100000)
	fmt.Println("Final theta:", theta)
	fmt.Println(predict(X, theta))

}
