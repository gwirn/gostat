package main

import (
	"errors"
	"fmt"
	"math"
)

/*
Creates a scaler to scale individual features to be in the range between 0 an 1

	:parameter
		* inSlice: the data that should be scaled where each vector represents one data point
	:return
		* scaler: function that scales a slice based on the minimum and maximum values of features in inSlice [(x-xmin)/(xmax-xmin)]
*/
func minMaxScaler(inSlice [][]float64) func([][]float64) {
	vectorSize := len(inSlice[0])
	// storage for the min and max values for each feature (column) 
	minVals := make([]float64, vectorSize)
	maxVals := make([]float64, vectorSize)
	for ci, i := range inSlice[0] {
		minVals[ci] = i
		maxVals[ci] = i
	}
	// find biggest / smallest values for each feature
	for _, i := range inSlice {
		for cj, j := range i {
			minVals[cj] = math.Min(minVals[cj], j)
			maxVals[cj] = math.Max(maxVals[cj], j)
		}
	}
	return func(sliceToScale [][]float64) {
		for ci, i := range inSlice {
			for cj := range i {
				sliceToScale[ci][cj] = (sliceToScale[ci][cj] - minVals[cj]) / (maxVals[cj] - minVals[cj])
			}
		}
	}
}

/*
Calculate the correlation coefficient for two (colIndX and colIndX) columns in inSlice

	:parameter
		*	inSlice: slice containing the data
		*	colIndX, colIndX: indices (zero indexed) of the columns for which the correlation should be computed
	:return
		*	corr: correlation coefficient between data in column colIndX and colIndY
*/
func corrCoef(inSlice [][]float64, colIndX, colIndY *int) (float64, error) {
	n := float64(len((inSlice)))
	sumX := 0.0
	sumY := 0.0
	sumXY := 0.0
	squareSumX := 0.0
	squareSumY := 0.0

	for _, i := range inSlice {
		Xi := i[*colIndX]
		Yi := i[*colIndY]
		sumX += Xi
		sumY += Yi
		sumXY += Xi * Yi
		squareSumX += Xi * Xi
		squareSumY += Yi * Yi
	}
	corr := (n*sumXY - sumX*sumY) / (math.Sqrt((n*squareSumX - sumX*sumX) * (n*squareSumY - sumY*sumY)))
	if math.IsNaN(corr) {
		return 0.0, errors.New(fmt.Sprintf("Couldn't calculate correlation between feature [%d] and [%d]", colIndX, colIndY))
	}
	return corr, nil
}
