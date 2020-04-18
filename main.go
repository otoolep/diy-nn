package main

import (
	"math"
	"math/rand"
	"time"
)

func main() {
	p := Perceptron{
		input:        [][]float64{{0, 0, 1}, {1, 1, 1}, {1, 0, 1}, {0, 1, 0}}, // Input Data
		actualOutput: []float64{0, 1, 1, 0},                                   // Actual Output
		epochs:       100000,                                                  // Number of Epoch
	}
	p.initialize()
	p.train()

	// Make Predictions
	print("Expected result is 0\n")
	print(p.forwardPass([]float64{0, 1, 0}), "\n")
	print(p.forwardPass([]float64{0, 1, 1}), "\n")
	print(p.forwardPass([]float64{0, 0, 0}), "\n")

	print("Expected result is 1\n")
	print(p.forwardPass([]float64{1, 1, 0}), "\n")
	print(p.forwardPass([]float64{1, 1, 1}), "\n")
	print(p.forwardPass([]float64{1, 0, 0}), "\n")
}

// Perceptron is a type of Neural Network.
type Perceptron struct {
	input        [][]float64
	actualOutput []float64
	weights      []float64
	bias         float64
	epochs       int
}

func (a *Perceptron) initialize() {
	rand.Seed(time.Now().UnixNano())
	a.bias = 0.0
	a.weights = make([]float64, len(a.input[0]))
	for i := 0; i < len(a.input[0]); i++ {
		a.weights[i] = rand.Float64()
	}
}

// sigmoid performs Sigmoid Activation
func (a *Perceptron) sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// forwardPass performs Forward Propagation
func (a *Perceptron) forwardPass(x []float64) (sum float64) {
	return a.sigmoid(dotProduct(a.weights, x) + a.bias)
}

// gradW calculates the Gradient of Weights
func (a *Perceptron) gradW(x []float64, y float64) []float64 {
	pred := a.forwardPass(x)
	return scalarMatMul(-(pred-y)*pred*(1-pred), x)
}

// gradB calculates the Gradients of Bias
func (a *Perceptron) gradB(x []float64, y float64) float64 {
	pred := a.forwardPass(x)
	return -(pred - y) * pred * (1 - pred)
}

// train trains the Perception for n epochs
func (a *Perceptron) train() {
	for i := 0; i < a.epochs; i++ {
		dw := make([]float64, len(a.input[0]))
		db := 0.0
		for length, val := range a.input {
			dw = vecAdd(dw, a.gradW(val, a.actualOutput[length]))
			db += a.gradB(val, a.actualOutput[length])
		}
		dw = scalarMatMul(2/float64(len(a.actualOutput)), dw)
		a.weights = vecAdd(a.weights, dw)
		a.bias += db * 2 / float64(len(a.actualOutput))
	}
}

// dotProduct returns the product of two vectors of the same size.
func dotProduct(v1, v2 []float64) float64 {
	dot := 0.0
	for i := 0; i < len(v1); i++ {
		dot += v1[i] * v2[i]
	}
	return dot
}

// vecAdd returns the addition of two vectors of the same size.
func vecAdd(v1, v2 []float64) []float64 {
	add := make([]float64, len(v1))
	for i := 0; i < len(v1); i++ {
		add[i] = v1[i] + v2[i]
	}
	return add
}

// scalarMatMul returns the product of a vector and a matrix.
func scalarMatMul(s float64, mat []float64) []float64 {
	result := make([]float64, len(mat))
	for i := 0; i < len(mat); i++ {
		result[i] += s * mat[i]
	}
	return result
}
