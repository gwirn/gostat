package main

import (
	"fmt"
	"log"
	"math"
)

/*
Finding the centroid of a given slice

	:paremeter
		*	inSlice: slice for which the centroid should be calculated
	:return
		*	centroidSlice: the centroid of the data
*/
func centroid(inSlice [][]float64) []float64 {
	centroidSlice := make([]float64, len(inSlice[0]))
	sampleNum := len(inSlice)
	for _, i := range inSlice {
		for cj, j := range i {
			centroidSlice[cj] += (j / float64(sampleNum))
		}
	}
	return centroidSlice
}

/*
finde the index of the smallest non- zero element in a slice

	:parameter
		*	inSlice: the slice to search in
	:return
		*	minIdx: index of the smallest non-zero element in the slice
*/
func argminNonZeroFloat(inSlice []float64) int {
	minIdx := 0
	minVal := inSlice[0]
	for ci, i := range inSlice {
		if i-0.0 > float64EqualityThreshold && i < minVal {
			minVal = i
			minIdx = ci
		}
	}
	return minIdx
}

/*
Hierarchical clustering using the centroids of each cluster for distance calculation

	:parameter
		*	inSlice: slice to be clusterd
		*	distType: which distance metric should be used
			-	euclidean
			-	manhattan
			-	hamming
			-	braycurtis
		*	maxIter: maximum number of iterations to find clusters
		*	maxDist: maximum distance between clusters to be allowed to merge
	:return
		*	cluster: indices of members of clusters in their own slice
*/
func hierachicalClustering(inSlice [][]float64, distType *string, maxIter *int, maxDist *float64) [][]int {
	// selecting the distance function
	distanceFunction := euclideanDist
	switch *distType {
	case "manhattan":
		distanceFunction = manhattanDist
	case "hamming":
		distanceFunction = hammingDist
	case "braycurtis":
		distanceFunction = braycurtisDiss
	case "euclidean":
		distanceFunction = euclideanDist
	default:
		fmt.Printf("Using default distance metric ['euclidean'] instead of the not implementd ['%s']", *distType)
		distanceFunction = euclideanDist
	}
	// storage for the indices of the clusters
	cluster := make([][]int, len(inSlice))
	for ci := range inSlice {
		cluster[ci] = []int{ci}
	}
	interCount := 0
	prevClusterNum := 0
	for {
		// calculate the centroids of each cluster
		centroidsOfCluster := make([][]float64, len(cluster))
		for ci, i := range cluster {
			clusterMembers := make([][]float64, len(i))
			for cj, j := range i {
				clusterMembers[cj] = inSlice[j]
			}
			centroidsOfCluster[ci] = centroid(clusterMembers)
		}
		// find the two clusters with the minimal distance between all available clusters
		minDist := math.Inf(1)
		partner1 := 0
		partner2 := 0
		for cj, j := range centroidsOfCluster {
			dist := distanceFunction(centroidsOfCluster, j)
			minDistIdx := argminNonZeroFloat(dist)
			if mDist := dist[minDistIdx]; mDist < minDist && minDistIdx != cj && mDist <= *maxDist {
				minDist = mDist
				partner1 = cj
				partner2 = minDistIdx
			}
		}
		if partner1 > 0 || partner2 > 0 {
			// merge cluster
			cluster[partner1] = append(cluster[partner1], cluster[partner2]...)
			// remove second partner of the merged cluster from the stored clusters since it's not in the merged clusters
			cluster = append(cluster[:partner2], cluster[partner2+1:]...)
			// stop if all are in on cluster or if the maxIter is reached
		}
		if len(cluster) == 1 || interCount == *maxIter || prevClusterNum == len(cluster) {
			break
		}
		prevClusterNum = len(cluster)
		interCount++
	}
	return cluster
}

/*
calculate the average correlation between two clusters

	:parameter
		*	inSlice: slice with data of features
		*	cluster1, cluster2: indices of feature in the same cluster
	:return
		*	totalCorr: the average correlation between the to clusters
*/
func ClusterCorrelation(inSlice [][]float64, cluster1, cluster2 []int) float64 {
	totalCorr := 0.0
	clusterMembers := 0
	for _, i := range cluster1 {
		for _, j := range cluster2 {
			if i != j {
				interCorr, err := corrCoef(inSlice, &i, &j)
				if err != nil {
					log.Fatal(fmt.Print("ClusterCorrelation couldn't be calculated", err))
				}
				totalCorr += math.Abs(interCorr)
				clusterMembers++
			}
		}
	}
	return totalCorr / float64(clusterMembers)
}

/*
Hierarchical clustering of features based on the mean correlation between all features in a cluster

	:parameter
		*	inSlice: slice to be clusterd
		*	maxIter: maximum number of iterations to find clusters
		*	minCorr: minimum correlation to be merged
	:return
		*	cluster: indices of members of clusters in their own slice
*/
func hierarchicalCorrelationClustering(inSlice [][]float64, maxIter *int, minCorr *float64) [][]int {
	// storage for the indices of the clusters
	cluster := make([][]int, len(inSlice[0]))
	for ci := range inSlice[0] {
		cluster[ci] = []int{ci}
	}

	prevClusterNum := len(cluster)
	interCount := 0
	for {
		// compare all clusters against each other and find the highest correlating
		maxCorr := 0.0
		partner1 := 0
		partner2 := 0
		for ci, i := range cluster {
			for cj, j := range cluster {
				if mCorr := ClusterCorrelation(inSlice, i, j); mCorr > maxCorr && ci != cj {
					maxCorr = mCorr
					partner1 = ci
					partner2 = cj
				}
			}
		}
		if partner1 > 0 || partner2 > 0 {
			// merge cluster
			cluster[partner1] = append(cluster[partner1], cluster[partner2]...)
			// remove second partner of the merged cluster from the stored clusters since it's not in the merged clusters
			cluster = append(cluster[:partner2], cluster[partner2+1:]...)
			// stop if all are in on cluster or if the maxIter is reached
		}
		if len(cluster) == 1 || interCount == *maxIter || prevClusterNum == len(cluster) {
			break
		}
		prevClusterNum = len(cluster)
		interCount++
	}
	return cluster
}
