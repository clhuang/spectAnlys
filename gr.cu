__constant__ int numFreqs;
__constant__ int numPoints;

__device__ int argMax(float* input) {
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

/**
  Output is numPoints * 3 array: height, center, standard deviation
  Input frequencies in nm to avoid underflow
*/
__global__ void gaussReg(float *input, float *output, float *frequencies, float cutoffPortion){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (idx >= numPoints) return;

    input += idx * numFreqs;
    output += idx * 3;

    float scaleFactor = 1e3 / (frequencies[numFreqs-1] - frequencies[0]);

    int maxI = argMax(input);

    float threshold = input[maxI] * cutoffPortion;

    float s40 = 0, s30 = 0, s20 = 0, s10 = 0, s00 = 0, s21 = 0, s11 = 0, s01 = 0; //quadratic regression stuff
    float logI, frequency;

    for (int i = maxI; i < numFreqs && input[i] > threshold; i++) {
       logI = __logf(input[i]);
       frequency = frequencies[i] - frequencies[maxI]; //shift over by maxFreq to avoid floating point errors
       frequency *= scaleFactor; //scale by scaleFactor to avoid underflow/overflow
       s00++;
       s10 += frequency;
       s20 += frequency * frequency;
       s30 += frequency * frequency * frequency;
       s40 += frequency * frequency * frequency * frequency;
       s01 += logI;
       s11 += frequency * logI;
       s21 += frequency * frequency * logI;
    }
    
    for (int i = maxI-1; i > 0 && input[i] > threshold; i--) {
       logI = __logf(input[i]);
       frequency = frequencies[i] - frequencies[maxI]; //shift over by maxFreq to avoid floating point errors
       frequency *= scaleFactor; //scale by scaleFactor to avoid underflow/overflow
       s00++;
       s10 += frequency;
       s20 += frequency * frequency;
       s30 += frequency * frequency * frequency;
       s40 += frequency * frequency * frequency * frequency;
       s01 += logI;
       s11 += frequency * logI;
       s21 += frequency * frequency * logI;
    }

    float a, b, c;
    //magical quadratic regression stuff
    a = (s21*(s20 * s00 - s10 * s10) - 
            s11*(s30 * s00 - s10 * s20) + 
            s01*(s30 * s10 - s20 * s20))
        /
        (s40*(s20 * s00 - s10 * s10) -
         s30*(s30 * s00 - s10 * s20) + 
         s20*(s30 * s10 - s20 * s20));

    b = (s40*(s11 * s00 - s01 * s10) - 
            s30*(s21 * s00 - s01 * s20) + 
            s20*(s21 * s10 - s11 * s20))
        /
        (s40 * (s20 * s00 - s10 * s10) - 
         s30 * (s30 * s00 - s10 * s20) + 
         s20 * (s30 * s10 - s20 * s20));

    c = (s40*(s20 * s01 - s10 * s11) - 
            s30*(s30 * s01 - s10 * s21) + 
            s20*(s30 * s11 - s20 * s21))
        /
        (s40 * (s20 * s00 - s10 * s10) - 
         s30 * (s30 * s00 - s10 * s20) + 
         s20 * (s30 * s10 - s20 * s20));

    //ax^2+bx+c=Ae^((x-b)^2/2c^2)

    if (a > 0) {
        output[0] = output[1] = output[2] = 0;
        return;
    }
    
    output[0] = __expf(c - (b * b) / (4 * a));
    output[1] = -b / scaleFactor / (2 * a) + frequencies[maxI];
    output[2] = sqrtf(-1 / (2 * a)) / scaleFactor;
    
}

__global__ void findPeaks(float *input, float *output, float *frequencies) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (idx >= numPoints) return;

    input = input + idx * numFreqs;
    output = output + idx;

    int maxI = argMax(input);

    output[0] = frequencies[maxI];
}

__device__ float integrate(float *input, float *frequencies, float lowerbound, float upperbound) {
    int i;
    float integral = 0, width, h1;

    for (i = 0; frequencies[i] < lowerbound && i < numFreqs; i++);
    if (i >= numFreqs) return 0;

    if (i != 0) {
        width = frequencies[i] - lowerbound;
        h1 = input[i-1] + (lowerbound - frequencies[i-1]) / (frequencies[i] - frequencies[i-1]) * (input[i] - input[i-1]);
        integral += (h1 + input[i]) * width / 2;
    }

    for(i++; frequencies[i] < upperbound && i < numFreqs; i++) {
        integral += (input[i] + input[i-1]) * (frequencies[i] - frequencies[i-1]) / 2;
    }

    if (i >= numFreqs) return integral;

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
        *output = integrate(input, frequencies, 0, targetFrequency);
    } else { //right
        *output = integrate(input, frequencies, targetFrequency, INFINITY);
    }
}

__global__ void integrateBounds(float *input, float *output, float *bounds, float *frequencies, bool left) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (idx >= numPoints) return;

    input = input + idx * numFreqs;
    output = output + idx;

    float lowerbound = bounds[idx];
    float upperbound = bounds[idx + numPoints];

    *output = integrate(input, frequencies, lowerbound, upperbound);
}
