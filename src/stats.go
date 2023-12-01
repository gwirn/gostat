package main

import (
	"errors"
	"fmt"
	"log"
	"math"
)

/*
Calculate the correlation coefficient for two (colIndX and colIndX) columns in inSlice

	:parameter
		* inSlice: slice containing the data
		* colIndX, colIndX: indices (zero indexed) of the columns for which the correlation should be computed
	:return
		* corr: correlation coefficient between data in column colIndX and colIndY
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

/*
Calculate the multiclass accuracy between prediction and ground truth

	:parameter
		* prediction: predicted labels as returned by a classifier
		* groundTruth: ground truth (correct) labels
	:return
		* acc: fraction of correctly classified samples
*/
func multiclassAccuracy(prediction, groundTruth []int) float64 {
	pSize := len(prediction)
	gTSize := len(groundTruth)
	if pSize != gTSize {
		log.Fatal(fmt.Printf("Prediction size [%d] doesn't match the ground truth size [%d]", pSize, gTSize))
	}
	correctClassification := 0
	for i := 0; i < pSize; i++ {
		if prediction[i] == groundTruth[i] {
			correctClassification++
		}
	}
	acc := float64(correctClassification) / float64(pSize)
	return acc
}
