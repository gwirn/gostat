package main

import (
	"fmt"
)

const float64EqualityThreshold = 1e-9

func main() {
	X := [][]float64{{1, 4}, {1, 3}, {1, 2}, {1, 1}, {1, 3}}
	y := []float64{3, 4, 5, 6, 7}
	theta := fit(X, y, 0.001, 1e-9, 100000)
	fmt.Println("Final theta:", theta)
	fmt.Println(predict(X, theta))

	/*
		x := [][]float64{{1, 2}, {3, 4}, {5, 6}}
		y := []float64{2, 3, 4}
		targ := []float64{7, 8}
		fmt.Println(euclideanDist(x, targ))
		fmt.Println(hammingDist(x, targ))
		fmt.Println(manhattanDist(x, targ))
		g := [][]float64{{6,7,4}}
		f := []float64{10,0,6}
		fmt.Println(braycurtisDiss(g, f))
		fmt.Println(kNNRegressor(x, y, targ, 3, "euclidean", true))
		yc := []int{2, 3, 4}
		fmt.Println(kNNClassifier(x, yc, targ, 3, "euclidean", true))
	*/
}
