#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cmath>
#include <string>
#include <cstdlib>
#include <limits>
#include <cmath>
#include <curand_kernel.h>
#include <cuda_runtime.h>

using namespace std;

struct DeviceData {
    int dimension;
    float* d_distance_preference;
    float* d_pheromone_preference;
    float* d_pheromone;
    float* d_distance_matrix;
    int* d_tours;
    bool* d_visited;
    float* d_tour_lengths;
    curandState* d_curandStates;
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
};

vector<float> flattenMatrix(const vector<vector<float>>& matrix) {
    int n = matrix.size();
    vector<float> flat;
    flat.reserve(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            flat.push_back(matrix[i][j]);
        }
    }
    return flat;
}

void findAndPrintBestTour(
    const DeviceData& deviceData,
    const std::string& outputFile
) {
    int n = deviceData.dimension;

    // Step 1: Download all tour lengths
    std::vector<float> tour_lengths(n);
    cudaMemcpy(tour_lengths.data(), deviceData.d_tour_lengths, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Step 2: Find the index of the best tour
    int bestAnt = -1;
    float bestLength = std::numeric_limits<float>::max();
    for (int i = 0; i < n; ++i) {
        if (tour_lengths[i] < bestLength) {
            bestLength = tour_lengths[i];
            bestAnt = i;
        }
    }

    // Step 3: Download the best tour
    std::vector<int> bestTour(n);
    cudaMemcpy(bestTour.data(), deviceData.d_tours + bestAnt * n, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Step 4: Find the index where city 1 appears
    int startIdx = -1;
    for (int i = 0; i < n; ++i) {
        if (bestTour[i] == 0) { // Remember cities are 0-based internally
            startIdx = i;
            break;
        }
    }

    if (startIdx == -1) {
        std::cerr << "Error: City 1 (index 0) not found in tour!" << std::endl;
        return;
    }

    // Step 5: Prepare rotated tour starting from city 1
    std::vector<int> rotatedTour;
    for (int i = 0; i < n; ++i) {
        rotatedTour.push_back(bestTour[(startIdx + i) % n]);
    }

    // Step 6: Output both to stdout and file
    std::cout.precision(10);
    std::cout << std::fixed << bestLength << "\n";
    for (int i = 0; i < n; ++i) {
        std::cout << (rotatedTour[i] + 1); // +1 to move from 0-based to 1-based
        if (i != n - 1) std::cout << " ";
    }
    std::cout << "\n";

    std::ofstream fout(outputFile);
    if (!fout) {
        std::cerr << "Error: could not open output file " << outputFile << std::endl;
        return;
    }

    fout.precision(10);
    fout << std::fixed << bestLength << "\n";
    for (int i = 0; i < n; ++i) {
        fout << (rotatedTour[i] + 1);
        if (i != n - 1) fout << " ";
    }
    fout << "\n";
    fout.close();
}

// Function to free all GPU-side resources
void freeDeviceData(DeviceData& deviceData) {
    cudaFree(deviceData.d_distance_preference);
    cudaFree(deviceData.d_pheromone_preference);
    cudaFree(deviceData.d_tours);
    cudaFree(deviceData.d_visited);
    cudaFree(deviceData.d_tour_lengths);
    cudaFree(deviceData.d_pheromone);
    cudaFree(deviceData.d_distance_matrix);
    cudaGraphDestroy(deviceData.graph);
    cudaGraphExecDestroy(deviceData.graphExec);
    cudaStreamDestroy(deviceData.stream);
}


// Parameters struct to store program parameters
struct Params {
    string inputFile;
    string outputFile;
    string type; // WORKER or QUEEN
    int numIter;
    float alpha;
    float beta;
    float evaporate;
    unsigned int seed;
};

// Function to compute Euclidean distance
float euclideanDistance(pair<float, float> a, pair<float, float> b) {
    return sqrt(powf(a.first - b.first, 2) + powf(a.second - b.second, 2));
}

// Parse command-line arguments into Params struct
Params parseArgs(int argc, char** argv, int dimension) {
    if (argc < 4) {
        cerr << "Usage: ./acotsp <input_file> <output_file> <TYPE> [NUM_ITER] [ALPHA] [BETA] [EVAPORATE] [SEED]" << endl;
        exit(1);
    }

    Params params;
    params.inputFile = argv[1];
    params.outputFile = argv[2];
    params.type = argv[3];

    // Set defaults
    params.numIter = dimension;
    params.alpha = 1.0;
    params.beta = 2.0;
    params.evaporate = 0.5;
    params.seed = 42; // Default seed if not provided

    // Parse optional arguments if provided
    if (argc > 4) params.numIter = atoi(argv[4]);
    if (argc > 5) params.alpha = atof(argv[5]);
    if (argc > 6) params.beta = atof(argv[6]);
    if (argc > 7) params.evaporate = atof(argv[7]);
    if (argc > 8) params.seed = (unsigned int)atoi(argv[8]);

    return params;
}

// Read cities from input file
vector<pair<float, float>> readCities(const string& filename, int& dimension) {
    ifstream fin(filename);
    if (!fin) {
        cerr << "Error opening input file: " << filename << endl;
        exit(1);
    }

    string line;
    dimension = 0;

    while (getline(fin, line)) {
        if (line.find("DIMENSION") != string::npos) {
            stringstream ss(line);
            string tmp;
            ss >> tmp >> tmp >> dimension;
        } else if (line.find("NODE_COORD_SECTION") != string::npos) {
            break;
        }
    }

    if (dimension <= 0) {
        cerr << "Invalid dimension in input file." << endl;
        exit(1);
    }

    vector<pair<float, float>> cities(dimension);

    for (int i = 0; i < dimension && getline(fin, line); ++i) {
        if (line == "EOF") break;
        stringstream ss(line);
        int cityNum;
        float x, y;
        ss >> cityNum >> x >> y;
        cities[cityNum - 1] = {x, y};
    }

    return cities;
}

// Calculate distance matrix
vector<vector<float>> calculateDistances(const vector<pair<float, float>>& cities) {
    int dimension = cities.size();
    vector<vector<float>> dist(dimension, vector<float>(dimension, 0.0));
    for (int i = 0; i < dimension; ++i) {
        for (int j = 0; j < dimension; ++j) {
            if (i != j) {
                dist[i][j] = euclideanDistance(cities[i], cities[j]);
            }
        }
    }
    return dist;
}

// Precompute the choice_info matrix, using BETA parameter
vector<vector<float>> calculateChoiceInfo(const vector<vector<float>>& distMatrix, float beta) {
    int dimension = distMatrix.size();
    vector<vector<float>> choice_info(dimension, vector<float>(dimension, 0.0));

    for (int i = 0; i < dimension; ++i) {
        for (int j = 0; j < dimension; ++j) {
            if (i != j && distMatrix[i][j] != 0.0) {
                choice_info[i][j] = powf(1.0 / distMatrix[i][j], beta);
            } else {
                choice_info[i][j] = 0.0; // Prevent division by zero and no self-loops
            }
        }
    }

    return choice_info;
}

__global__ void initCurandStatesKernel(curandState* states, unsigned int seed, int n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;

    curand_init(seed + k * 1000ULL, 0, 0, &states[k]);
}

__global__ void constructToursWorker(
    const float* distance_preference,
    const float* pheromone_preference,
    int* tours,
    bool* visited,
    curandState* states,
    int n
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;

    // Load curandState
    curandState state = states[k];

    int startCity = curand(&state) % n;
    tours[k * n] = startCity;

    for (int i = 0; i < n; ++i) {
        visited[k * n + i] = false;
    }
    visited[k * n + startCity] = true;

    for (int step = 1; step < n; ++step) {
        int currentCity = tours[k * n + step - 1];

        float sum_probs = 0.0;
        for (int j = 0; j < n; ++j) {
            if (!visited[k * n + j]) {
                sum_probs += distance_preference[currentCity * n + j] * pheromone_preference[currentCity * n + j];
            }
        }

        float r = curand_uniform(&state) * sum_probs;

        float cumulative_prob = 0.0;
        int nextCity = n - 1;
        for (int j = 0; j < n; ++j) {
            if (!visited[k * n + j]) {
                cumulative_prob += distance_preference[currentCity * n + j] * pheromone_preference[currentCity * n + j];
                if (cumulative_prob >= r) {
                    nextCity = j;
                    break;
                }
            }
        }

        tours[k * n + step] = nextCity;
        visited[k * n + nextCity] = true;
    }

    // Save updated curandState back
    states[k] = state;
}

__global__ void evaluateTours(
    const int* tours,               // Tours: num_ants x n
    const float* distance_matrix,   // Distances between cities: n x n (flattened)
    float* tour_lengths,            // Output: total length per tour
    int n
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return; // Assuming 1 thread per ant

    float length = 0.0;

    for (int step = 0; step < n - 1; ++step) {
        int from = tours[k * n + step];
        int to   = tours[k * n + step + 1];
        length += distance_matrix[from * n + to];
    }

    // Return to the start city
    int last = tours[k * n + n - 1];
    int first = tours[k * n];
    length += distance_matrix[last * n + first];

    tour_lengths[k] = length;
}

__global__ void evaporatePheromones(
    float* pheromone,
    float evaporate,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;

    pheromone[idx] *= (1.0 - evaporate);
}

__global__ void depositPheromones(
    const int* tours,
    const float* tour_lengths,
    float* pheromone,
    int n
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return; // One thread per ant

    float contribution = 1.0 / tour_lengths[k];

    // Add contribution for each edge along the tour (excluding return)
    for (int step = 0; step < n - 1; ++step) {
        int from = tours[k * n + step];
        int to   = tours[k * n + step + 1];

        atomicAdd(&pheromone[from * n + to], contribution);
    }

    // Handle the return to starting city separately
    int last = tours[k * n + n - 1];
    int first = tours[k * n + 0];

    atomicAdd(&pheromone[last * n + first], contribution);
}


__global__ void updatePheromonePreference(
    const float* pheromone,
    float* pheromone_preference,
    float alpha,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;

    pheromone_preference[idx] = powf(pheromone[idx], alpha);
}

DeviceData prepareDeviceData(
    const vector<vector<float>>& distance_preference,
    const vector<vector<float>>& distance_matrix,
    float alpha,
    float beta,
    float evaporate,
    unsigned int seed
) {
    DeviceData deviceData;
    deviceData.dimension = distance_preference.size();
    int n = deviceData.dimension;

    // --- Flatten distance_preference ---
    vector<float> distance_preference_flat;
    distance_preference_flat.reserve(n * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            distance_preference_flat.push_back(distance_preference[i][j]);

    // --- Flatten distance_matrix ---
    vector<float> dist_flat;
    dist_flat.reserve(n * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            dist_flat.push_back(distance_matrix[i][j]);

    // --- Allocate device memory ---
    cudaMalloc(&deviceData.d_distance_preference, n * n * sizeof(float));
    cudaMalloc(&deviceData.d_pheromone_preference, n * n * sizeof(float));
    cudaMalloc(&deviceData.d_pheromone, n * n * sizeof(float));
    cudaMalloc(&deviceData.d_distance_matrix, n * n * sizeof(float));

    cudaMalloc(&deviceData.d_tours, n * n * sizeof(int));
    cudaMalloc(&deviceData.d_visited, n * n * sizeof(bool));
    cudaMalloc(&deviceData.d_tour_lengths, n * sizeof(float));

    // Allocate curand states
    cudaMalloc(&deviceData.d_curandStates, n * sizeof(curandState));

    // --- Upload initial data ---
    cudaMemcpy(deviceData.d_distance_preference, distance_preference_flat.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceData.d_distance_matrix, dist_flat.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);

    // --- Properly initialize pheromones ---
    vector<float> ones(n * n, 1.0);
    cudaMemcpy(deviceData.d_pheromone, ones.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceData.d_pheromone_preference, ones.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);

    // --- Initialize curand states ---
    int blockSizeInit = 128;
    int gridSizeInit = (n + blockSizeInit - 1) / blockSizeInit;
    initCurandStatesKernel<<<gridSizeInit, blockSizeInit>>>(deviceData.d_curandStates, seed, n);
    cudaDeviceSynchronize(); // ensure curand states are ready

    // --- Setup CUDA Graph ---
    int blockSizeEdges = 128;
    int gridSizeEdges = (n * n + blockSizeEdges - 1) / blockSizeEdges;

    int blockSizeAnts = 128;
    int gridSizeAnts = (n + blockSizeAnts - 1) / blockSizeAnts;

    cudaStreamCreate(&deviceData.stream);
    cudaStreamBeginCapture(deviceData.stream, cudaStreamCaptureModeGlobal);

    // 1. Construct tours
    constructToursWorker<<<gridSizeAnts, blockSizeAnts, 0, deviceData.stream>>>(
        deviceData.d_distance_preference,
        deviceData.d_pheromone_preference,
        deviceData.d_tours,
        deviceData.d_visited,
        deviceData.d_curandStates, // <- pass curand states
        n
    );

    // 2. Evaluate tour lengths
    evaluateTours<<<gridSizeAnts, blockSizeAnts, 0, deviceData.stream>>>(
        deviceData.d_tours,
        deviceData.d_distance_matrix,
        deviceData.d_tour_lengths,
        n
    );

    // 3. Evaporate pheromones
    evaporatePheromones<<<gridSizeEdges, blockSizeEdges, 0, deviceData.stream>>>(
        deviceData.d_pheromone,
        evaporate,
        n
    );

    // 4. Deposit new pheromones
    depositPheromones<<<gridSizeAnts, blockSizeAnts, 0, deviceData.stream>>>(
        deviceData.d_tours,
        deviceData.d_tour_lengths,
        deviceData.d_pheromone,
        n
    );

    // 5. Update pheromone preference
    updatePheromonePreference<<<gridSizeEdges, blockSizeEdges, 0, deviceData.stream>>>(
        deviceData.d_pheromone,
        deviceData.d_pheromone_preference,
        alpha,
        n
    );

    cudaStreamEndCapture(deviceData.stream, &deviceData.graph);
    cudaGraphInstantiate(&deviceData.graphExec, deviceData.graph, nullptr, nullptr, 0);

    return deviceData;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: ./acotsp <input_file> <output_file> <TYPE> [NUM_ITER] [ALPHA] [BETA] [EVAPORATE] [SEED]" << std::endl;
        return 1;
    }

    // 1. Parse input parameters
    Params params = parseArgs(argc, argv, 0); // We fix dimension after reading cities

    // 2. Read city coordinates and distance matrix
    int dimension = 0;
    std::vector<std::pair<float, float>> cities = readCities(params.inputFile, dimension);
    params.numIter = (argc > 4) ? atoi(argv[4]) : dimension; // default if needed

    // 3. Precompute distance matrix
    std::vector<std::vector<float>> distMatrix = calculateDistances(cities);

    // 4. Precompute distance preference matrix (1/dist)^BETA
    std::vector<std::vector<float>> distancePreference = calculateChoiceInfo(distMatrix, params.beta);

    // 5. Prepare device data, CUDA memory, CUDA Graph
    DeviceData deviceData = prepareDeviceData(
        distancePreference,
        distMatrix,
        params.alpha,
        params.beta,
        params.evaporate,
        params.seed
    );

    // 6. Create CUDA timing events
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent, deviceData.stream);

    // 7. Main ACO iteration loop
    for (int iter = 0; iter < params.numIter; ++iter) {
        cudaGraphLaunch(deviceData.graphExec, deviceData.stream);
        cudaStreamSynchronize(deviceData.stream);

        if (iter % 100 == 0 || iter == params.numIter - 1) {
            std::cout << "Iteration " << iter + 1 << " / " << params.numIter << std::endl;
        }
    }

    cudaEventRecord(stopEvent, deviceData.stream);
    cudaEventSynchronize(stopEvent);

    // 8. Measure elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);
    std::cout << "Total ACO time: " << milliseconds << " ms" << std::endl;

    // 9. Find and print best tour
    findAndPrintBestTour(deviceData, params.outputFile);

    // 10. Cleanup
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    freeDeviceData(deviceData);

    return 0;
}

