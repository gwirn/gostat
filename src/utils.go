package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
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
/*
Generate training data from a give csv file and scale it 
	:parameter
		* filePath: path to the file 
		* testFrac: how much of the data should be used for testing (between 0 and 1)
		* catConv: true to convert categorical data to integer labels for the labels - not needed when labels are already integers in the csv
	:return
		* trainDSFeatures: training features
		* trainDSLabel: training labels
		* testDSFeatures: test features
		* testDSLabel: test labels
		* labelMap: map to convert that was used to convert string labels to int labels
		* &scaler: the scaler function used to scale the data
		*/
func genTrainTestData(filePath *string, testFrac *float64, catConv *bool) ([][]float64, []int, [][]float64, []int, map[string]int, *func([][]float64)) {
	// read raw csv
	lines := readCsvFile(*filePath)
	// number of lines in the csv
	numLines := len(lines)
	// number of entries in the line
	lineSize := len(lines[0])
	// number of features per sample
	numFeatures := lineSize -1
	// stored labels
	labels := make([]string, numLines)
	uniqueLabels := []string{}
	// stored feature vector
	features := make([][]float64, numLines)
	for ci, i := range lines {
		// converted features if line i
		lineConv := make([]float64, numFeatures)
		for j := 0; j < lineSize; j++ {
			if j == 0 {
				// add to labels
				labels[ci] = i[j]
				if !isinString(uniqueLabels, i[j]) {
					uniqueLabels = append(uniqueLabels, i[j])
				}
			} else {
				// convert all feature vector entries to float
				convFloat, err := strconv.ParseFloat(i[j], 64)
				if err != nil {
					log.Fatal(fmt.Sprintf("Couldn't convert [%s] at line [%d] to float64\n", i[j], ci), err)
				}
				lineConv[j-1] = convFloat
			}
		}
		features[ci] = lineConv
	}
	// map to change string labels to int
	labelMap := make(map[string]int)
	for ci, i := range uniqueLabels {
		if *catConv {
			labelMap[i] = ci
		} else {
			// if labels are already integers in the csv
			iConv, err := strconv.Atoi(i)
			if err != nil {
				log.Fatal(fmt.Sprintf("Couldn't convert [%s] to int\n", i), err)
			}
			labelMap[i] = iConv
		}
	}
	// change string labels to int labels
	labelsInt := make([]int, numLines)
	for ci, i := range labels {
		labelsInt[ci] = labelMap[i]
	}
	// scale all features to be within 0, 1
	scaler := minMaxScaler(features)
	scaler(features)
	// randomly shuffle the dataset
	shuffleDatasetFloatString(features, labelsInt)
	// split the dataset
	border := int(float64(numLines) * *testFrac)
	trainDSFeatures, trainDSLabel := features[:border], labelsInt[:border]
	testDSFeatures, testDSLabel := features[border:], labelsInt[border:]
	return trainDSFeatures, trainDSLabel, testDSFeatures, testDSLabel, labelMap, &scaler
}
