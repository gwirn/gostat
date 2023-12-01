package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
)

/*
Shuffle data set in place

	:parameter
		* featureSlice: features describing the data
		* labelSlice: labels for the data
	:return
		None
*/
func shuffleDatasetFloatString(featureSlice [][]float64, labelSlice []int) {
	fSize := len(featureSlice)
	lSize := len(labelSlice)
	if fSize != lSize {
		log.Fatal(fmt.Printf("Prediction size [%d] doesn't match the ground truth size [%d]", fSize, lSize))
	}
	rand.Shuffle(fSize, func(i, j int) {
		featureSlice[i], featureSlice[j] = featureSlice[j], featureSlice[i]
		labelSlice[i], labelSlice[j] = labelSlice[j], labelSlice[i]
	})
}

/*
Test whether a string is in inSlice or not

	:parameter
		* inSlice: the slice to be tested
		* target: the string to be tested whether it is in inSlice or not
	:retun
		* isin: true if target is in inSlice
*/
func isinString(inSlice []string, target string) bool {
	isin := false
	for _, i := range inSlice {
		if i == target {
			isin = true
			break
		}
	}
	return isin
}

/*
Helper function to read csv file

	:parameter
		* filePath: path to the csv file to be read
	:return
		* records: lines of the csv file
*/
func readCsvFile(filePath string) [][]string {
	f, err := os.Open(filePath)
	if err != nil {
		log.Fatal("Unable to read input file "+filePath, err)
	}
	defer f.Close()

	csvReader := csv.NewReader(f)
	records, err := csvReader.ReadAll()
	if err != nil {
		log.Fatal("Unable to parse file as CSV for "+filePath, err)
	}

	return records
}

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
