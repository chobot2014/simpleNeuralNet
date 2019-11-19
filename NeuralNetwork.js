
const x = [[0,0,1],[0,1,1],[1,0,1],[1,1,1]];
const y = [[0],[1],[1],[0]];


var groupBy = function(xs, key) {
    return xs.reduce(function(rv, x) {
      (rv[x[key]] = rv[x[key]] || []).push(x);
      return rv;
    }, {});
  };

function sigmoid(t) {
    if (typeof t === 'number') {
        return 1/(1+Math.pow(Math.E, -t));
    } else if (t.length) {
        return t.map(function(x) {
            return sigmoid(x);
        })
    }    
}

function sigmoidDerivative(p) {
    if (p.length) {
        let output = [];
        for(let i=0; i< p.length; i++) {
            output.push(sigmoidDerivative(p[i]));
        }
        return output;
    } else {
        return p * (1-p);
    }    
}

function calculateLoss(real, predicted) {        
    let output = [];    
    for(let i =0; i< real.length; i++) {
        output.push(2*(real[i] = predicted[i]) * sigmoidDerivative(predicted[i]));
    }
    return output;
}


function plusEqualArray(arr1, arr2) {    
    for (let i =0; i < arr1.length; i++) {
        for (let j= 0; j < arr1.length; j++) {
            arr1[i][j] += arr2[i][j];
        }
    }
    return arr1;
}


function dot(a,b) {
    // returns the dot product of two arrays. 
    // For 2-D vectors, it is the equivalent to matrix multiplication. 
    // For 1-D arrays, it is the inner product of the vectors. 
    // For N-dimensional arrays, it is a sum product over the last axis of a and the second-last axis of b.

    function timesAllTheNumbersInTheThingByTheOtherNumber(theThing, theOtherNumber, out) {
        for (let i =0; i< theThing.length; i++) {
            if (theThing[i].length) {
                out[i] = [];
                timesAllTheNumbersInTheThingByTheOtherNumber(theThing[i], theOtherNumber, out[i]);
            } else {
                out.push(theThing[i] * theOtherNumber);
            }
        }                
    }


    if (typeof a === 'number' && typeof b === 'number') {
        return a * b;
    } else {
        if (a.length && b.length) {            
            let output = [];
            for(let i=0; i < a.length; i++) {                
                for (let j = 0; j < a[i].length; j++) {                                        
                    for (let k = 0; k < b.length; k++) {                        
                        output.push({iBy: i, kBy: k, xBy: a[i][j], val: a[i][j]*b[j][k]});                   
                    }                    
                }                            
            }           
            let groups = output.reduce(function(acc, curr, i) {
                if (acc.find(function(x){return x.find(function(y){ return y.iBy === curr.iBy})})) {
                    acc[acc.findIndex(function(x){return x.find(function(y){ return y.iBy === curr.iBy})})].push(curr);
                } else {
                    acc.push([curr]);
                }
                return acc;
            }, []).map(function(x,i) {
                let highestKBy= Math.max(...x.map(a => a.kBy));
                let highestXBy = Math.max(...x.map(a => a.xBy));
                let out = [];
                for (let k = 0; k <= highestKBy; k++) {        
                    let kItem =0;
                    for (let r = 0; r <= highestXBy; r++) {
                        let foundX = x.find(function(x){ return x.kBy === k && x.xBy ===r});
                        if (foundX) {
                            kItem += foundX.val;
                        }                        
                    }
                    out.push(kItem)
                }
                return out;
            });
            
            return groups;
        } else if (a.length && !b.length) {
            let output = [];            
            timesAllTheNumbersInTheThingByTheOtherNumber(a,b, output);            
            return output;
        } else if (!a.length && b.length) {
            let output = [];            
            timesAllTheNumbersInTheThingByTheOtherNumber(b,a, output);
            return output;
        }
    }
}


function pivotArray(arr) { //arr: Array<Array<number>>    
    const output = [];    
    for (let i=0; i< arr.length; i++) {    
        for (let j=0; j < arr[i].length; j++) {
            if (typeof output[j] !== 'undefined') {
                output[j].push(arr[i][j]);
            } else {
                output.push([arr[i][j]]);
            }            
        }        
    }
    return output;
}


class NeuralNetwork {
    constructor(x, y) {
        this.input = x;
        this.y = y;        
        this.weights1 = [];
        for(let i = 0; i < this.input[1].length; i++) {            
            this.weights1.push([Math.random(), Math.random(), Math.random(), Math.random()]);
        }
        
        this.weights2 = [[Math.random()],[Math.random()],[Math.random()]];        
        
        this.output = [];
        for(let i=0; i < y.length; i++) {
            this.output.push([0]);
        }        
    }

    feedforward() {               
        this.layer1 = sigmoid(dot(this.input, this.weights1));        
        this.layer2 = sigmoid(dot(this.layer1, this.weights2));        
        return this.layer2;        
    }

    backprop(){        
        let dWeights2 = dot(pivotArray(this.layer1), calculateLoss(this.y.slice(), this.output));
        let dWeights1 = dot(
            pivotArray(this.input), 
            dot(
                calculateLoss(this.y.slice(), this.output), 
                pivotArray(this.weights2)
            ).map(function(x,i){ return x*sigmoidDerivative(this.layer1.slice())[i]})
        );                
        
        this.weights1 = plusEqualArray(this.weights1.slice(), dWeights1);        
        this.weights2 = plusEqualArray(this.weights2.slice(), dWeights2);        
        
    }

    train() {
        this.output = this.feedforward();
        this.backprop();
    }
}


let nn = new NeuralNetwork(x,y);

for (let i=0; i < 1500; i++) {
    if (i % 100 == 0) {
        console.log("Actual Output: \n" + y)
        console.log("Predicted Output: \n" + nn.feedforward())        
        console.log("\n")
    }
    nn.train();
}