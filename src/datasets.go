package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
)

/*
Helper function to read csv file

	:parameter
		* filePath: path to the csv file to be read
		* header: true if there is a header
	:return
		* headLine: header of the file
		* records: lines of the csv file
*/
func readCsvFile(filePath *string, header *bool) ([]string, [][]string) {
	f, err := os.Open(*filePath)
	if err != nil {
		log.Fatalln(fmt.Sprintf("Unable to open input file [%s]\n", *filePath), err)
	}
	defer f.Close()

	csvReader := csv.NewReader(f)
	headLine := []string{}
	// read header
	if *header {
		firstLine, err := csvReader.Read()
		if err != nil {
			log.Fatalln(fmt.Sprintf("Couldn't read header of [%s]\n", *filePath), err)
		}
		headLine = append(headLine, firstLine...)
	}
	records, err := csvReader.ReadAll()
	if err != nil {
		log.Fatalln(fmt.Sprintf("Unable to parse file as CSV for [%s]\n", *filePath), err)
	}

	return headLine, records
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
		* firstLineLabels: true if the first line in the csv file is a header
		* useScaler: true to scale the features to be within the range of 0 to 1
	:return
		* trainDSFeatures: training features
		* trainDSLabel: training labels
		* testDSFeatures: test features
		* testDSLabel: test labels
		* labelMap: map to convert that was used to convert string labels to int labels
		* &scaler: the scaler function used to scale the data
		*/
func genTrainTestData(filePath *string, testFrac *float64, catConv *bool, firstLineLabels *bool, useScaler *bool) ([][]float64, []int, [][]float64, []int, map[string]int, *func([][]float64)) {
	// read raw csv
	_, lines := readCsvFile(filePath, firstLineLabels)
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
				if len(i[j]) == 0 {
					lineConv[j-1] = math.NaN()
				} else {
					convFloat, err := strconv.ParseFloat(i[j], 64)
					if err != nil {
						log.Fatalln(fmt.Sprintf("Couldn't convert [%s] at line [%d] to float64\n", i[j], ci), err)
					}
					lineConv[j-1] = convFloat
				}
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
				log.Fatalln(fmt.Sprintf("Couldn't convert [%s] to int\n", i), err)
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
	if *useScaler {
		scaler(features)
	}
	// randomly shuffle the dataset
	shuffleDatasetFloatString(features, labelsInt)
	// split the dataset
	border := int(float64(numLines) * *testFrac)
	trainDSFeatures, trainDSLabel := features[:border], labelsInt[:border]
	testDSFeatures, testDSLabel := features[border:], labelsInt[border:]
	return trainDSFeatures, trainDSLabel, testDSFeatures, testDSLabel, labelMap, &scaler
}

/*
Search for features that do not change and create a new csv only containing non constant features
	:parameter
		* oldFilePath: path to the original csv file
		* newFilePath: path to the new csv file
		* header: whether a header should is in the old file
	:return
		* constantFeatures: indices of columns that are constant
		* newSlice: inSlice with removed constant columns
*/
func nonConstantCSV(oldFilePath, newFilePath *string, header *bool) {
	oldHeader, oldCSV := readCsvFile(oldFilePath, header)
	// number of data points int the slice
	sliceSize := len(oldCSV)
	// number of features per data point
	numFeatures := len(oldCSV[0])
	constantFeatures := []int{}
	notConstantFeatures := []int{}
	// slice containing the non constant data
	newSlice := make([][]string, sliceSize)
	for f:=0;f<numFeatures;f++{
		firstVal := oldCSV[0][f]
		constant := true
		for i:=1;i<sliceSize;i++{
			// if a different entry to the first entry is found -> not constant
			if oldCSV[i][f] != firstVal {
				constant = false
			}
		} 
		if constant {
			constantFeatures = append(constantFeatures, f)
		} else {
			notConstantFeatures = append(notConstantFeatures, f)
			for cj, j := range oldCSV {
				newSlice[cj] = append(newSlice[cj], j[f])
			}
		}
	}
	
	// header of the non constant features
	newHeader := make([]string, len(notConstantFeatures))
	for ci, i := range notConstantFeatures {
		newHeader[ci] = oldHeader[i]
	}
	// create a file
    file, err := os.Create(*newFilePath)
    if err != nil {
        log.Fatalln(fmt.Sprintf("Couldn't create file at [%s]\n", *newFilePath), err)
    }
	// write to file
    defer file.Close()
	writer := csv.NewWriter(file)
	writer.Write(newHeader)
	writer.WriteAll(newSlice)
	defer writer.Flush()
	if err := writer.Error(); err != nil {
		log.Fatalln(fmt.Sprintf("Couldn't write to csv at [%s]\n", *newFilePath), err)
	}
	fmt.Println("**log**")
	fmt.Printf("From [%d] [%d] features were removed\nRemoved features:\n", numFeatures, len(constantFeatures))
	for _, i := range constantFeatures {
		fmt.Printf("%s, ", oldHeader[i])
	}
}
