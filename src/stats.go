package main

import (
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
func corrCoef(inSlice [][]float64, colIndX, colIndY *int) float64 {
	n := float64(len((inSlice)))
	sumX := 0.0
	sumY := 0.0
	sumXY := 0.0
	squareSumX := 0.0
	squareSumY := 0.0

	minX, maxX := inSlice[0][*colIndX],inSlice[0][*colIndX]
	minY, maxY := inSlice[0][*colIndY],inSlice[0][*colIndY]
	for _, i := range inSlice {
		Xi := i[*colIndX]
		Yi := i[*colIndY]
		minX = math.Min(minX, Xi)
		maxX = math.Max(maxX, Xi)
		minY = math.Min(minY, Yi)
		maxY = math.Max(maxY, Yi)
		sumX += Xi
		sumY += Yi
		sumXY += Xi * Yi
		squareSumX += Xi * Xi
		squareSumY += Yi * Yi
	}
	if math.Abs(minX-maxX) < float64EqualityThreshold {
		log.Fatalln(fmt.Sprintf("Feature [%d] is constant - correlation calculation is not possible", *colIndX))
	}
	if math.Abs(minY - maxY) < float64EqualityThreshold {
		log.Fatalln(fmt.Sprintf("Feature [%d] is constant - correlation calculation is not possible", *colIndY))
	}
	corr := (n*sumXY - sumX*sumY) / (math.Sqrt((n*squareSumX - sumX*sumX) * (n*squareSumY - sumY*sumY)))
	if math.IsNaN(corr) {
		log.Fatalln(fmt.Sprintf("Couldn't calculate correlation between feature [%d] and [%d]", colIndX, colIndY))
	}
	return corr
}

/*
Calculate the multi class accuracy between prediction and ground truth

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
		log.Fatal(fmt.Printf("Prediction size [%d] doesn't match the ground truth size [%d]\n", pSize, gTSize))
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

/*
Calculate the mean absolute error

	:parameter
		* prediction: predicted labels as returned by a classifier
		* groundTruth: ground truth (correct) labels
	:return
		* mae: mean absolute error
*/
func mae(prediction, groundTruth []float64) *float64 {
	pSize := len(prediction)
	if gTSize := len(groundTruth); pSize != gTSize {
		log.Fatal(fmt.Printf("Prediction size [%d] doesn't match the ground truth size [%d]\n", pSize, gTSize))
	}
	sumError := 0.0
	for i := 0; i < pSize; i++ {
		sumError += math.Abs(prediction[i] - groundTruth[i])
	}
	mae := sumError / float64(pSize)
	return &mae
}

/*
Calculate the mean squared error

	:parameter
		* prediction: predicted labels as returned by a classifier
		* groundTruth: ground truth (correct) labels
	:return
		* mse: mean squared error
*/
func mse(prediction, groundTruth []float64) *float64 {
	pSize := len(prediction)
	if gTSize := len(groundTruth); pSize != gTSize {
		log.Fatal(fmt.Printf("Prediction size [%d] doesn't match the ground truth size [%d]\n", pSize, gTSize))
	}
	sumError := 0.0
	for i := 0; i < pSize; i++ {
		sumError += math.Pow(prediction[i]-groundTruth[i], 2)
	}
	mae := sumError / float64(pSize)
	return &mae
}
