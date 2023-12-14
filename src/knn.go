package main

import (
	"fmt"
	"log"
	"math"
	"sort"
)

/*
Sort by generating an array of indices that index data of inSlice in sorted order

	:parameter
		*	inSlice: the slice to be sorted
	:return
		*	indices: the indices that sort the inSlice
*/
func argsort(inSlice []float64) []int {
	indices := make([]int, len(inSlice))
	for i := range indices {
		indices[i] = i
	}
	sort.Slice(indices, func(i, j int) bool {
		return inSlice[indices[i]] < inSlice[indices[j]]
	})
	return indices
}

/*
Calculating the Euclidean distance [sqrt(sum((x - y)^2))] between a set of vecorts x and another vector target

	:parameter
		*	x: set of vectors against which the distance should be computed
		*	target: vector for which the distances should be computed
	:return
		*	dist: all distances between x and target
*/
func euclideanDist(x [][]float64, target []float64) []float64 {
	xSize := len(x)
	dist := make([]float64, xSize)
	for ci, i := range x {
		iDist := 0.0
		for cj, j := range target {
			iDist += math.Pow(i[cj]-j, 2)
		}
		dist[ci] = math.Sqrt(iDist)
	}
	return dist
}

/*
Calculating the Hamming distance [N_unequal(x, y) / N_tot] between a set of vecorts x and another vector target

	:parameter
		*	x: set of vectors against which the distance should be computed
		*	target: vector for which the distances should be computed
	:return
		*	dist: all distances between x and target
*/
func hammingDist(x [][]float64, target []float64) []float64 {
	xSize := len(x)
	vectorSize := len(x[0])
	dist := make([]float64, xSize)
	for ci, i := range x {
		iDist := 0.0
		for cj, j := range target {
			if math.Abs(i[cj]-j) >= float64EqualityThreshold {
				iDist++
			}
		}
		dist[ci] = iDist / float64(vectorSize)
	}
	return dist
}

/*
Calculating the Manhattan distance [sum(|x - y|)] between a set of vecorts x and another vector target

	:parameter
		*	x: set of vectors against which the distance should be computed
		*	target: vector for which the distances should be computed
	:return
		*	dist: all distances between x and target
*/
func manhattanDist(x [][]float64, target []float64) []float64 {
	xSize := len(x)
	dist := make([]float64, xSize)
	for ci, i := range x {
		iDist := 0.0
		for cj, j := range target {
			iDist += math.Abs(i[cj] - j)
		}
		dist[ci] = iDist
	}
	return dist
}

/*
Calculating the Bray-Curtis dissimilarity [sum(|x - y|) / (sum(|x|) + sum(|y|))] between a set of vecorts x and another vector target

	:parameter
		*	x: set of vectors against which the distance should be computed
		*	target: vector for which the distances should be computed
	:return
		*	diss: all dissimilarity between x and target
*/
func braycurtisDiss(x [][]float64, target []float64) []float64 {
	xSize := len(x)
	diss := make([]float64, xSize)
	for ci, i := range x {
		cij := 0.0
		iSum := 0.0
		jSum := 0.0
		for cj, j := range target {
			iAbs := math.Abs(i[cj])
			jAbs := math.Abs(j)
			iSum += iAbs
			jSum += jAbs
			cij += math.Abs(i[cj] - j)
		}
		diss[ci] = cij / (iSum + jSum)
	}
	return diss
}

/*
Random forest regressor

	:parameter
		*	x: vectors representing the training data
		*	y: values of the training data
		*	target: vector of the data for which y should be predicted
		*	k: number of samples used for the prediction
		*	distType: which distance metric should be used
			-	euclidean
			-	manhattan
			-	hamming
			-	braycurtis
		*	scaleDist: whether to scale the prediction based on the distance of samples to the target
	:return
		*	result: regression result
		*	sortedDistIdx: slice with indices sorting the distances/ values from small to big
		*	dists: distances to all samples in x
*/
func kNNRegressor(x [][]float64, y []float64, target []float64, k *int, distType *string, scaleDist *bool) (float64, []int, []float64) {
	if xSize, ySize := len(x), len(y); xSize != ySize {
		log.Fatal(fmt.Printf("Size of x [%d] not equal to size of y [%d]", xSize, ySize))
	}
	// calc distance
	dists := []float64{}
	switch *distType {
	case "manhattan":
		dists = append(dists, manhattanDist(x, target)...)
	case "hamming":
		dists = append(dists, hammingDist(x, target)...)
	case "braycurtis":
		dists = append(dists, braycurtisDiss(x, target)...)
	case "euclidean":
		dists = append(dists, euclideanDist(x, target)...)
	default:
		dists = append(dists, euclideanDist(x, target)...)
		fmt.Printf("Using default distance metric ['euclidean'] instead of the not implementd ['%s']\n", *distType)
	}
	// sort distances small to big
	sortedDistIdx := argsort(dists)
	// nearest neighbours y values
	nnYs := make([]float64, *k)
	// nearest neighbours distances
	nnDists := make([]float64, *k)
	for i := 0; i < *k; i++ {
		nnYs[i] = y[sortedDistIdx[i]]
		nnDists[i] = dists[sortedDistIdx[i]]
	}

	// calculate the result
	result := 0.0
	if *scaleDist {
		// sum of distances of the k neighbours
		distSum := 0.0
		// scale the impact based on the distance (closer equals higher impact)
		for ci, i := range nnYs {
			weight := 0.0
			if nnDists[ci] <= float64EqualityThreshold {
				weight = 1
			} else {
				weight = 1 / nnDists[ci]
			}
			distSum += weight
			result += i * weight
		}
		result = result / distSum
	} else {
		for _, i := range nnYs {
			result += i
		}
		result = result / float64(*k)
	}
	return result, sortedDistIdx, dists
}

/*
Random forest classifier

	:parameter
		*	x: vectors representing the training data
		*	y: classes of the training data
		*	target: vector of the data for which y should be predicted
		*	k: number of samples used for the prediction
		*	distType: which distance metric should be used
			-	euclidean
			-	manhattan
			-	hamming
			-	braycurtis
		*	scaleDist: whether to scale the prediction based on the distance of samples to the target
	:return
		*	result: regression result
		*	sortedDistIdx: slice with indices sorting the distances/ values from small to big
		*	dists: distances to all samples in x
		*	resultClasses: percentages for all classes
*/
func kNNClassifier(x [][]float64, y []int, target []float64, k *int, distType *string, scaleDist *bool) (*int, []int, []float64, map[int]float64) {
	if xSize, ySize := len(x), len(y); xSize != ySize {
		log.Fatal(fmt.Printf("Size of x [%d] not equal to size of y [%d]\n", xSize, ySize))
	}
	// calc distance
	dists := []float64{}
	switch *distType {
	case "manhattan":
		dists = append(dists, manhattanDist(x, target)...)
	case "hamming":
		dists = append(dists, hammingDist(x, target)...)
	case "braycurtis":
		dists = append(dists, braycurtisDiss(x, target)...)
	case "euclidean":
		dists = append(dists, euclideanDist(x, target)...)
	default:
		dists = append(dists, euclideanDist(x, target)...)
		fmt.Printf("Using default distance metric ['euclidean'] instead of the not implementd ['%s']\n", *distType)
	}
	// sort distances small to big
	sortedDistIdx := argsort(dists)
	// nearest neighbours y values
	nnYs := make([]int, *k)
	// nearest neighbours distances
	nnDists := make([]float64, *k)
	for i := 0; i < *k; i++ {
		nnYs[i] = y[sortedDistIdx[i]]
		nnDists[i] = dists[sortedDistIdx[i]]
	}

	// calculate the result
	resultClasses := make(map[int]float64)
	if *scaleDist {
		// sum of distances of the k neighbours
		distSum := 0.0
		// scale the impact based on the distance (closer equals higher impact)
		for ci, i := range nnYs {
			weight := 0.0
			if nnDists[ci] == 0 {
				weight = 1
			} else {
				weight = 1 / nnDists[ci]
			}
			distSum += weight
			resultClasses[i] += weight
		}
		for ci := range resultClasses {
			resultClasses[ci] = resultClasses[ci] / distSum
		}
	} else {
		// add a fraction per sample to its class
		fractSample := 1 / float64(*k)
		for _, i := range nnYs {
			resultClasses[i] += fractSample
		}
	}
	// find the class with the highest percentage
	result := -1
	resultPercent := math.Inf(-1)
	for key, value := range resultClasses {
		if value > resultPercent {
			result = key
			resultPercent = value
		}
	}
	return &result, sortedDistIdx, dists, resultClasses
}
