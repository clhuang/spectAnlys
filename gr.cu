__constant__ int numFreqs;
__constant__ int numPoints;
__constant__ float centerFreq;

__device__ int argmax(float* input) {
    float max = input[0];
    int best = 0;
    for (int i = 1; i < numFreqs; i++) {
        if (input[i] > max) {
            max = input[i];
            best = i;
        }
    }

    return best;
}

__device__ int dCountPeaks(float *input, float *frequencies) {
    int peaks = 0;
    for (int i = 1; i < numFreqs - 1; i++) {
        if ((input[i] > input[i-1] || (input[i] == input[i-1] && i > 1 && input[i] > input[i-2]))
                && input[i] > input[i+1]) peaks++;
    }

    return peaks;
}

/*Output is numPoints * 3 array: height, center, standard deviation (a*e^(-(x-b)^2/(2c^2)))
  If quadratic is true, outputs an approximation for the top of the gaussian in the form a(x-b)^2 + c*/
__global__ void reg(float *input, float *output, float *frequencies, float cutoffPortion, bool quadratic){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (idx >= numPoints) return;

    input += idx * numFreqs;
    output += idx * 3;

    float scaleFactor = 1e3 / (frequencies[numFreqs-1] - frequencies[0]);

    int maxI = argmax(input);

    float threshold = input[maxI] * cutoffPortion;

    float s40 = 0, s30 = 0, s20 = 0, s10 = 0, s00 = 0, s21 = 0, s11 = 0, s01 = 0; //quadratic regression stuff
    float yval, frequency;

    //go left from peak, then go right from peak
    for (int i = maxI; i < numFreqs && input[i] > threshold; i++) {
       yval = quadratic ? input[i] / input[maxI] : __logf(input[i]); //in quadratic, divide by input[maxI] to avoid underflow
       frequency = frequencies[i] - frequencies[0]; //shift over by maxFreq to avoid floating point errors
       frequency *= scaleFactor; //scale by scaleFactor to avoid underflow/overflow
       s00++;
       s10 += frequency;
       s20 += frequency * frequency;
       s30 += frequency * frequency * frequency;
       s40 += frequency * frequency * frequency * frequency;
       s01 += yval;
       s11 += frequency * yval;
       s21 += frequency * frequency * yval;
    }
    
    for (int i = maxI-1; i > 0 && input[i] > threshold; i--) {
       yval = quadratic ? input[i] / input[maxI] : __logf(input[i]);
       frequency = frequencies[i] - frequencies[0]; //shift over by maxFreq to avoid floating point errors
       frequency *= scaleFactor; //scale by scaleFactor to avoid underflow/overflow
       s00++;
       s10 += frequency;
       s20 += frequency * frequency;
       s30 += frequency * frequency * frequency;
       s40 += frequency * frequency * frequency * frequency;
       s01 += yval;
       s11 += frequency * yval;
       s21 += frequency * frequency * yval;
    }

    float a, b, c, q;
    //magical quadratic regression stuff
    q = (s40*(s20 * s00 - s10 * s10) -
         s30*(s30 * s00 - s10 * s20) + 
         s20*(s30 * s10 - s20 * s20));
    a = (s21*(s20 * s00 - s10 * s10) - 
            s11*(s30 * s00 - s10 * s20) + 
            s01*(s30 * s10 - s20 * s20))
        / q;

    b = (s40*(s11 * s00 - s01 * s10) - 
            s30*(s21 * s00 - s01 * s20) + 
            s20*(s21 * s10 - s11 * s20))
        / q;

    c = (s40*(s20 * s01 - s10 * s11) - 
            s30*(s30 * s01 - s10 * s21) + 
            s20*(s30 * s11 - s20 * s21))
        / q;

    //ax^2+bx+c=Ae^((x-b)^2/2c^2)

    if (a != a || b != b || c != c) {
        output[0] = output[1] = output[2] = 0;
        return;
    }
    if (a > 0) {
        output[0] = output[1] = output[2] = -1;
    }

    if (quadratic) { //scalefactors and input[maxI] and frequencies[0] terms are there to fix scaled/shifted values
        output[0] = a * input[maxI] / scaleFactor;
        output[1] = -b / scaleFactor / (2 * a) + frequencies[0];
        output[2] = (c - b * b / (4 * a)) * input[maxI];
    } else {
        output[0] = __expf(c - (b * b) / (4 * a));
        output[1] = -b / scaleFactor / (2 * a) + frequencies[0];
        output[2] = sqrtf(-1 / (2 * a)) / scaleFactor;
    }
}

/*Finds the absolute peak values of the function.*/
__global__ void findPeaks(float *input, float *output, float *frequencies) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (idx >= numPoints) return;

    input = input + idx * numFreqs;
    output = output + idx;

    int maxI = argmax(input);

    output[0] = frequencies[maxI];
}

/*Counts the number of peaks, where d2y/df2 < 0 and dy/df=0*/
__global__ void countPeaks(float *input, int *output, float *frequencies) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (idx >= numPoints) return;

    input += idx * numFreqs;
    output += idx;

    *output = dCountPeaks(input, frequencies);
}

__device__ float integrate(float *input, float *frequencies, float lowerbound, float upperbound) {
    int i;
    float integral = 0, width, h1;

    for (i = 0; frequencies[i] < lowerbound && i < numFreqs; i++);
    if (i >= numFreqs) return 0;

    if (i != 0) {
        //interpolation to find trapezoid size
        width = frequencies[i] - lowerbound;
        h1 = input[i-1] + (lowerbound - frequencies[i-1]) / (frequencies[i] - frequencies[i-1]) * (input[i] - input[i-1]);
        integral += (h1 + input[i]) * width / 2;
    }

    for(i++; frequencies[i] < upperbound && i < numFreqs; i++) {
        integral += (input[i] + input[i-1]) * (frequencies[i] - frequencies[i-1]) / 2;
    }

    if (i >= numFreqs) return integral;

    //interpolation to find trapezoid size
    width = upperbound - frequencies[i-1];
    h1 = input[i-1] + (upperbound - frequencies[i-1]) / (frequencies[i] - frequencies[i-1]) * (input[i] - input[i-1]);
    integral += (h1 + input[i-1]) * width / 2;

    return integral;
}

__global__ void integrateLR(float *input, float *output, float *target, float *frequencies, bool left) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (idx >= numPoints) return;

    input = input + idx * numFreqs;
    output = output + idx;

    float targetFrequency = frequencies[idx];
    
    if (left) {
        *output = integrate(input, frequencies, -INFINITY, targetFrequency);
    } else { //right
        *output = integrate(input, frequencies, targetFrequency, INFINITY);
    }
}

__global__ void integrateBounds(float *input, float *output, float *bounds, float *frequencies) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (idx >= numPoints) return;

    input = input + idx * numFreqs;
    output = output + idx;

    float lowerbound = bounds[idx];
    float upperbound = bounds[idx + numPoints];

    *output = integrate(input, frequencies, lowerbound, upperbound);
}

/* Finds the width of the profile at half maximum.*/
__global__ void fwhm(float *input, float *output, float *frequencies) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (idx >= numPoints) return;

    input = input + idx * numFreqs;
    output = output + idx;

    int peak = argmax(input);
    int halfIntensity = input[peak];
    float upperbound;
    float lowerbound;

    int i;
    for (i = peak; input[i] > halfIntensity && i < numFreqs; i++);
    upperbound = frequencies[i-1] + (frequencies[i] - frequencies[i-1]) *
        (halfIntensity - input[i]) / (input[i-1] - input[i]);
    for (i = peak; input[i] > halfIntensity && i >= 0; i--);
    lowerbound = frequencies[i] + (frequencies[i+1] - frequencies[i]) *
        (halfIntensity - input[i]) / (input[i+1] - input[i]);

    *output = upperbound - lowerbound;
}
