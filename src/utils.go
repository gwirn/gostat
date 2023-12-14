package main

import (
	"fmt"
	"log"
	"math/rand"
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
		log.Fatal(fmt.Printf("Feature size [%d] doesn't match the label size [%d]", fSize, lSize))
	}
	rand.Shuffle(fSize, func(i, j int) {
		featureSlice[i], featureSlice[j] = featureSlice[j], featureSlice[i]
		labelSlice[i], labelSlice[j] = labelSlice[j], labelSlice[i]
	})
}

/*
Calculate the sum of entries in inSlice
	:parameter
		* inSlice: slice that should be summed up
	:return
		* sum: sum of the slice
*/
func sumFloat64(inSlice []float64) *float64 {
	sum := 0.0
	for _, i := range inSlice {
		sum += i
	}
	return &sum
}

/*
Check two float64 slices for equal length

	:parameter
		* inSlice1, inSlice2: the slices to be compared
	:return
		None
*/
func assertEqualLengthFloat(inSlice1, inSlice2 []float64){
	if l1, l2 := len(inSlice1), len(inSlice2); l1 != l2 {
		log.Fatal(fmt.Printf("First slice [len %d] has not the same size as the second slice [len %d]", l1, l2))
	}
}

/*
Check two int slices for equal length

	:parameter
		* inSlice1, inSlice2: the slices to be compared
	:return
		None
*/
func assertEqualLengthInt(inSlice1, inSlice2 []int){
	if l1, l2 := len(inSlice1), len(inSlice2); l1 != l2 {
		log.Fatal(fmt.Printf("First slice [len %d] has not the same size as the second slice [len %d]", l1, l2))
	}
}

/*
Test whether a int is in inSlice or not

	:parameter
		* inSlice: the slice to be tested
		* target: the string to be tested whether it is in inSlice or not
	:retun
		* isin: true if target is in inSlice
*/
func isinInt(inSlice []int, target int) bool {
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
