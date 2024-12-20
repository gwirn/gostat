package main

import (
	"fmt"
	"sync"
)

const float64EqualityThreshold = 1e-9

func main() {
	// rand.Seed(42)
	// read data csv train KNN and testt accuracy

	// fPath := "/Volumes/SanDisk/cac_pointcloud_data.csv"
	fPath := "../datasets/TUANDROMDnoCor.csv"
	// fPath := "../datasets/TUANDROMDnew.csv"
	// nFPath := "../datasets/TUANDROMDnoCor.csv"
	// header := true
	// nonConstantCSV(&fPath, &nFPath, &header)
	// fPath := "../datasets/spambase/spambaseNew.data"
	// fPath := "../datasets/iris/dataNew.csv"
	k := 1
	distanceMetric := "euclidean"
	trainFract := 0.8
	convertCat := true
	firstLineLabels := true
	scaleFeatures := false
	// whether the neighbor importance should be scaled by distance
	scale := false

	trainFeatures, trainLabels, testFeatures, testLabels, _, _ := genTrainTestData(&fPath, &trainFract, &convertCat, &firstLineLabels, &scaleFeatures)
	testSize := len(testLabels)
	pred := make([]int, testSize)
	var wg sync.WaitGroup
	wg.Add(testSize)
	for ci, i := range testFeatures {
		go func(ci int, i []float64) {
			var err error
			res, _, _, _ := kNNClassifier(trainFeatures, trainLabels, i, &k, &distanceMetric, &scale)
			pred[ci] = *res
			if err != nil {
				panic(err)
			}
			wg.Done()
		}(ci, i)
	}
	wg.Wait()
	fmt.Println(multiclassAccuracy(testLabels, pred))

	/*
		for i := 0; i < testSize; i++ {
			res, _, _, _ := kNNClassifier(trainFeatures, trainLabels, testFeatures[i], &k, &distanceMetric, &scale)
			pred[i] = *res
		}
	*/
	/*
		// cluster correlating attributes
		maximumIteration := 20
		minimumCorrelation := .6
		clusters := hierarchicalCorrelationClustering(trainFeatures, &maximumIteration, &minimumCorrelation)
		trainSize := len(trainFeatures)
		newTrainFeatures := make([][]float64, trainSize)
		newTestFeatures := make([][]float64, testSize)
		for _, i := range clusters {
			feature := -1
			if len(i) > 1 {
					feature = *findRepresentativeForCluster(trainFeatures, i)
			} else {
				feature =  i[0]
			}
			for cj, j := range trainFeatures {
				newTrainFeatures[cj] = append(newTrainFeatures[cj], j[feature])
			}
			for cj, j := range testFeatures {
				newTestFeatures[cj] = append(newTestFeatures[cj], j[feature])
			}
		}
	*/
	/*
		// read data csv train KNN and testt accuracy
		fPath := "../datasets/spambase/spambaseNew.data"
		trainFract := 0.8
		convertCat := false
		trainFeatures, trainLabels, testFeatures, testLabels, _, _ := genTrainTestData(&fPath, &trainFract, &convertCat)
		testSize := len(testLabels)
		pred := make([]int, testSize)
		k := 10
		distanceMetric := "euclidean"
		scale := true
		for i := 0; i < testSize; i++ {
			res, _, _, _ := kNNClassifier(trainFeatures, trainLabels, testFeatures[i], &k, &distanceMetric, &scale)
			pred[i] = res
		}
		fmt.Println(multiclassAccuracy(testLabels, pred))
		// cluster hierarchically based on distance
		data := [][]float64{{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}, {7, 7}, {8, 8}, {9, 9}}
		distanceType := "euclidean"
		maximumIterations := 100
		maximumDistance := 7.
		fmt.Println(hierachicalClustering(data, &distanceType, &maximumIterations, &maximumDistance))

		// cluster based on correlation
		data := [][]float64{{1,2,0}, {2,3,1}, {3,4,1}, {4,5,0}}
		maximumIteration := 10
		minimumCorrelation := .2
		fmt.Println(hierarchicalCorrelationClustering(data, &maximumIteration, &minimumCorrelation))

		// correlation coefficient
		x := [][]float64{{15,25}, {18,25}, {21,27}, {24,31}, {27,32}}
		x = [][]float64{{43, 99}, {21, 65}, {25, 79}, {42, 75}, {57, 87}, {59, 81}}
		fmt.Println(corrCoef(x, 0, 1))

		// get scaler to scale the data
		x := [][]float64{{1, 2}, {3, 4}, {5, 6}}
		scaler := minMaxScaler(x)
		scaler(x)
		fmt.Println(x)

		x := [][]float64{{1, 2}, {3, 4}, {5, 6}}
		y := []float64{2, 3, 4}
		targ := []float64{7, 8}
		fmt.Println(euclideanDist(x, targ))
		fmt.Println(hammingDist(x, targ))
		fmt.Println(manhattanDist(x, targ))
		g := [][]float64{{6,7,4}}
		f := []float64{10,0,6}
		fmt.Println(braycurtisDiss(g, f))
		numK := 3
		distanceType := "euclidean"
		scale := true
		fmt.Println(kNNRegressor(x, y, targ, &numK, &distanceType, &scale))
		yc := []int{2, 3, 4}
		fmt.Println(kNNClassifier(x, yc, targ, &numK, &distanceType, &scale))
	*/
}
