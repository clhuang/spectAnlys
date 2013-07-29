__constant__ int numFreqs;
__constant__ int numPoints;

/**
  Output is numPoints * 3 array: height, center, standard deviation
  Input frequencies in nm to avoid underflow
*/
__global__ void gaussReg(float *input, float *output, float *frequencies, float cutoffPortion){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (idx >= numPoints) return;

    input += idx * numFreqs;
    output += idx * 3;

    float threshold = input[0];
    float maxFreq = frequencies[0];
    float scaleFactor = 1e3 / (frequencies[numFreqs-1] - frequencies[0]);
    for (int i = 0; i < numFreqs; i++) {
        if (input[i] > threshold) {
            threshold = input[i];
            maxFreq = frequencies[i];
        }
    }

    threshold *= cutoffPortion;

    float s40 = 0, s30 = 0, s20 = 0, s10 = 0, s00 = 0, s21 = 0, s11 = 0, s01 = 0; //quadratic regression stuff
    float logI, frequency;

    for (int i = 0; i < numFreqs; i++) {
        if (input[i] > threshold) {
           logI = __logf(input[i]);
           frequency = frequencies[i] - maxFreq; //shift over by maxFreq to avoid floating point errors
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
    output[1] = -b / scaleFactor / (2 * a) + maxFreq;
    output[2] = sqrtf(-1 / (2 * a)) / scaleFactor;
    
}

__global__ void findPeaks(float *input, float *output, float *frequencies) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (idx >= numPoints) return;

    input = input + idx * numFreqs;
    output = output + idx;

    float maxIntensity = input[0];
    float maxFrequency = 0;

    for (int i = 0; i < numFreqs; i++) {
        if (input[i] > maxIntensity){
            maxIntensity = input[i];
            maxFrequency = frequencies[i];
        }
    }

    output[0] = maxFrequency;
}

__global__ void integrate(float *input, float *output, float *target, float *frequencies, bool left) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (idx >= numPoints) return;

    input = input + idx * numFreqs;
    output = output + idx;

    float targetFrequency = frequencies[idx];
    float integral = 0;
    
    if (left) {
        for (int i = 1; frequency[i] < targetFrequency; i++) {
            integral += (output[i] + output[i-1]) * (frequency[i] - frequency[i-1]) / 2;
        }
    } else { //right
        for (int i = numFreqs - 1; frequency[i] > targetFrequency; i++) {
            integral += (output[i] + output[i+1]) * (frequency[i+1] - frequency[i]) / 2;
        }
    }

    output[0] = integral;
}
