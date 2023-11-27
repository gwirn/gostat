package main

import (
	"fmt"
	"log"
	"math"
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
		updatedTheta[j] = (theta[j] - alpha*gradient) / float64(m) * 2
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
		*	deltaThetaTH: how much coefficients need to change per update in order to continue 1e-9
		*	nIter: number of iterations for gradient decent (100000)
	:return:
		*	theta: estimated coefficients for linear regression
*/
func fit(X [][]float64, y []float64, alpha, deltaThetaTH float64, nIter int) []float64 {
	ySize := len(y)
	if xSize := len(X); xSize != ySize {
		log.Fatal(fmt.Printf("Sizes of X [%d] and y [%d] don't match'", xSize, ySize))
	}
	// Initialize model theats
	predSize := len(X[0])
	theta := make([]float64, predSize)
	ySizeFloat := float64(ySize)

	// Train the model using gradient descent
	for i := 0; i < nIter; i++ {
		iTheta := GradientDescent(X, y, theta, alpha)
		// how many coefficients did change less than deltaThetaTH since last update
		thMet := 0
		for j := range iTheta {
			if math.Abs(theta[j]-iTheta[j]) < deltaThetaTH {
				thMet++
			}
			theta[j] = iTheta[j]
		}
		// compute loss
		if i%1000 == 0 {
			diff := 0.0
			for cl, l := range predict(X, theta) {
				diff += math.Abs(l - y[cl])
			}
			fmt.Printf("Iteration %-6d-- MAE: %f \n", i, diff/ySizeFloat)
		}
		if thMet == predSize {
			fmt.Println("Stopped at iteration: ", i)
			break
		}
	}
	return theta
}
