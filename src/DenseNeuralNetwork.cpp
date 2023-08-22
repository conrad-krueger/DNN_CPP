#include "DenseNeuralNetwork.h"
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <random>       // std::default_random_engine
#include <chrono>       // std::chrono::system_clock
#include <fstream>
#include<algorithm>

using std::cout , std::endl, std::string, std::vector;


/**
 * Constructor that creates the Dense Neural Network
 * @param loss name of loss function (only supports mse and cross entropy)
 * @param numOfNodes vector of integers that describe how many nodes are in each layer (index 0 describes how many nodes in first layer)
 * @param activationPerLayer vector of string that contain the names of the activation functions on each corresponding inner layer
 * @update weights, bias, nodes, weightTotal, nodeTotal, layerTotal, numOfNodes
 */
DenseNeuralNetwork::DenseNeuralNetwork(string loss, vector<size_t> numOfNodes, vector<string> activationPerLayer): delta(), loss(loss), weights(), nodes(), biases(),layerTotal(numOfNodes.size()), nodeTotal(), weightTotal(), nodePerLayer(numOfNodes), activationPerLayer(activationPerLayer), L(), D_L(), activation(), activationDerivatives() {

    if (numOfNodes.size() < 2){
        throw "ERROR: Dense Neural Network should contain atleast 2 layers";
    }

    srand(time(NULL));

    if (loss == "mean_squared_error"){
        L = mse;
        D_L = Dmse;
    }
    else if (loss == "cross_entropy"){
        L = cross_entropy;
        D_L = Dcross_entropy;
    }else {
        throw "ERROR: Dense Neural Network Class version 2.0 only accepts \"mean_squared_error\" or \"cross_entropy\" as loss functions";
    }
        
    if (activationPerLayer.size()+1 != numOfNodes.size())
        throw "ERROR: Activation Layers should have n-1 elements where n is the number of layers in the network";
    
    //populate vectors containing functions
    for (string active : activationPerLayer){
        if(active == "logistic"){ 
            activationDerivatives.push_back(Dlogistic);
            activation.push_back(logistic);

        }else if(active == "relu"){
            activationDerivatives.push_back(Drelu);
            activation.push_back(relu);

        }else if(active == "tanh"){
            activationDerivatives.push_back(Dtanh);
            activation.push_back(tanh);

        }else if(active == "softmax"){
            activationDerivatives.push_back(Dsoftmax);
            activation.push_back(linear); //will account for later
            
        }else if(active == "linear"){
            activationDerivatives.push_back(Dlinear);
            activation.push_back(linear);

        }else{
            throw "ERROR: Must use on of the following Activation Functions: relu, logistic, softmax, tanh, or linear";
        }
    }
     
    //allocate delta for updating 
    delta = new double*[layerTotal-1]; 

    for (size_t i = layerTotal-1; i > 0; i--)
        delta[i-1] = new double[nodePerLayer.at(i)]{};


    //weights[connected section index][number of node in next layer][number of node in prev layer]

    weights = new double**[numOfNodes.size()-1]; //has n-1 connecting sections
    for (size_t i = 0; i < numOfNodes.size()-1; i++){

        weights[i] = new double*[numOfNodes.at(i+1)];

        for (size_t j = 0; j < numOfNodes[i+1]; j++){
            
            // Prevents oversaturation in the beginning I set each of the weights to have a value in the range of 0 +/- 1/sqrt(number of nodes in receiving layer)
            weights[i][j] = new double[numOfNodes[i]];
            for (size_t k = 0; k < numOfNodes[i]; k++){
                weights[i][j][k] = -1/sqrt(numOfNodes[i+1]) + ((double)rand() / RAND_MAX) * (1/sqrt(numOfNodes[i+1])*2); 
                weightTotal++;
            }
        }
    }

    // Init nodes && activation functions for each layer (default is ReLU and last is logistic)
    nodes = new double*[numOfNodes.size()];

    for (size_t i = 0; i < numOfNodes.size(); i++){
        nodes[i] = new double[numOfNodes[i]]{};
        nodeTotal += numOfNodes[i];
    }
     
    // Init biases (all to 0)
    biases = new double[numOfNodes.size()-1]{};
    for (size_t i = 0; i < numOfNodes.size()-1; i++){
        biases[i] = -1/sqrt(numOfNodes[i+1]) + ((double)rand() / RAND_MAX) * (1/sqrt(numOfNodes[i+1])*2);
    }
}


/**
 * Copies another DenseNeuralNetwork object's weights and biases into this object
 * @param dnn DenseNeuralNetwork object to be copied
 * @return None
 */
void DenseNeuralNetwork::copy(const DenseNeuralNetwork& dnn){
    clear();

    this->loss = dnn.getLossFunction();

    //new stuff
    activation = dnn.getActivations();
    activationDerivatives = dnn.getActivationsDerivatives();

    if (loss == "mean_squared_error"){
        L = mse;
        D_L = Dmse;
    }
    else if (loss == "cross_entropy"){
        L = cross_entropy;
        D_L = Dcross_entropy;
    }else {
        throw "ERROR: Dense Neural Network Class version 2.0 only accepts \"mean_squared_error\" or \"cross_entropy\" as loss functions";
    }

    //Copy delta
    delta = new double*[dnn.getNodePerLayer().size()-1]; 
    for (size_t i = dnn.getNodePerLayer().size()-1; i > 0; i--)
        delta[i-1] = new double[dnn.getNodePerLayer().at(i)]{};    

    //weights[connected section index][number of node in next layer][number of node in prev layer]
    weights = new double**[dnn.getNodePerLayer().size()-1]; //has n-1 connecting sections
    for (size_t i = 0; i < dnn.getNodePerLayer().size()-1; i++){

        weights[i] = new double*[dnn.getNodePerLayer().at(i+1)];

        for (size_t j = 0; j < dnn.getNodePerLayer().at(i+1); j++){
            
            weights[i][j] = new double[dnn.getNodePerLayer().at(i)];
            for (size_t k = 0; k < dnn.getNodePerLayer().at(i); k++){
                weights[i][j][k] = dnn.getWeight(i,j,k);
                
            }
        }
    }
    weightTotal = dnn.getWeightTotal();


    //init nodes && activation functions for each layer (default is ReLU and last is logistic)
    nodes = new double*[dnn.getNodePerLayer().size()];

    for (size_t i = 0; i < dnn.getNodePerLayer().size(); i++){
        nodes[i] = new double[dnn.getNodePerLayer().at(i)]{}; //no need to copy nodes values, will change based on input
    }
    nodeTotal = dnn.getNodeTotal();

     
    //init biases (all to 0)
    biases = new double[dnn.getNodePerLayer().size()-1]{};
    for (size_t i = 0; i < dnn.getNodePerLayer().size()-1; i++){
        biases[i] = dnn.getBias(i);
    }


    layerTotal = dnn.getNodePerLayer().size();
    nodePerLayer = dnn.getNodePerLayer();

    activationPerLayer = dnn.getActivationStr();

}


/**
 * Clear the contents on this DenseNeuralNetwork object (deallocate all malloc memory)
 * @update nodes, weights, biases, nodes, layerTotal, nodeTotal, weightTotal, nodePerLayer, activationPerLayer
 * @return None
 */
void DenseNeuralNetwork::clear(){
    loss = "";

    //deallocate delta
    if (delta){
        for (size_t i = layerTotal-1; i > 0; i--)
            delete[] delta[i-1];
        delete[] delta;
        delta = nullptr;
    }
    
    //deallocate weight matricies
    if (weights){

        for (size_t i = 0; i < layerTotal-1; i++){
            for (size_t j = 0; j < nodePerLayer.at(i+1); j++){
                delete[] weights[i][j];
            }
            delete[] weights[i];
        }
        delete[] weights;

        weights = nullptr;
        weightTotal = 0;

    }
    
    //deallocate node matricies
    if (nodes){
        for(size_t i = 0; i < layerTotal; i++){
            delete[] nodes[i];
        }
        delete[] nodes;
        nodes = nullptr;
        nodeTotal = 0;
    }
    


    //deallocate bias array
    if (biases){
        delete[] biases;
        biases = nullptr;
    }
    

    layerTotal = 0;

    nodePerLayer = {};

    activationPerLayer = {};

    activation = {};
    activationDerivatives = {};

    L = nullptr;
    D_L = nullptr;
}



/**
 * Prints out the values of every node
 * @return None
 */
void DenseNeuralNetwork::printNodes(){
     for (size_t i = 0; i < layerTotal; i++){
        for (size_t j = 0; j < nodePerLayer.at(i); j++){
             cout << nodes[i][j] << " ";
        }
        cout << endl;
     }
}

/**
 * Prints out information about the DNN Architecture, including the weight matricies, the biases, number of layers.
 * @param explain boolean value that explains how to read weight matricies
 * @return None
 */
void DenseNeuralNetwork::info(){
    cout << "==========================================================" << endl;
    cout << "Dense Neural Network Information:" << endl;


    cout << "Total Layers: " << layerTotal << "\t Total Nodes: " << nodeTotal << "\t Total Weights: "<< weightTotal << "\t Total Biases:"<< layerTotal-1 << endl << endl; 
    
    //Weight Matricies are read as matrix[i][j] = weight from node j to node i

    for (size_t i = 0; i < layerTotal-1; i++){
        cout << "Weight Matrix from Layer " << i << " [" << nodePerLayer.at(i) << " Node(s)] --> Layer "<<  i+1 << " [" << nodePerLayer.at(i) << " Node(s)]" << endl;
        cout << "----------------------------------" << endl;

        for (size_t j = 0; j < nodePerLayer.at(i+1); j++){

            //init all to 1
            for (size_t k = 0; k < nodePerLayer.at(i); k++){
               cout << weights[i][j][k] << " ";
            }
            cout << endl;

        }
        cout << "\n+ " << biases[i] << " (bias at Layer "<< i+1 << ")" << endl << endl << endl; 

    }
    cout << "==========================================================" << endl <<endl;

}

/**
 * Loads DNN Architecture with custom weights and biases
 * @param weights 1D vector that contains weights for each node connection. Weights should be ordered sequentially (first layer, then second layer, etc.). Weights pointing to the same node must be local to each other in memory. NOTE: Look at EX for better understanding of locality.
 * EX: Input layers: node a, b , c; Hidden Layer: node x, y; Output Layer: node z; Weights: [A -> X, B -> X, C -> X, A -> Y, B -> Y, C -> Y, X -> Z, Y -> Z] 
 * @param biases 1D vector that contains the bias amount for each layer. Should be ordered sequentially.
 * @updates weights, biases
 * @return None
 */
void DenseNeuralNetwork::loadWB(vector<double> weights, vector<double> biases){ //sofar good
    if (biases.size() != layerTotal-1 || weights.size() != weightTotal){
        string desc = "ERROR: Incorrect amount of weights and/or biases\nNeed ";
        desc += std::to_string(layerTotal-1) + " biases (Given: " +  std::to_string(biases.size()) + ")\nNeed ";
        desc += std::to_string(weightTotal) + " weights (Given: " +  std::to_string(weights.size()) + ")";
        throw desc;
    }
    
    //load weights 
    int currLoading = 0;
    for (size_t i = 0; i < layerTotal-1; i++){
        for (size_t j = 0; j < nodePerLayer.at(i+1); j++){
            for (size_t k = 0; k < nodePerLayer.at(i); k++){
               this->weights[i][j][k] = weights.at(currLoading);
               currLoading++;
            }
        }
    }

    //load biases
    for (size_t i = 0; i < layerTotal-1; i++){
        this->biases[i] = biases.at(i);
    }

}


/**
 * Saves Model information into a .txt file
 * @param filename name of txt file
 * @return None
 */
void DenseNeuralNetwork::save(string filename){ 
    //check if file exists
    std::ifstream check(filename);
    if (check){
        cout << "WARNING: File already exists. Do you want to override it? (y/n) ";
        char indicate;
        std::cin >> indicate;
        if (indicate != 'y')
            return;
    }
    
    std::ofstream out; 
    out.open(filename);

    if (! out.is_open())
        throw "ERROR: File could not be opened";

    //loss
    out << "loss: " << loss << "\n";
    //layers
    out << "layers: ";
    for (size_t nodes: nodePerLayer)
        out << nodes << " ";
    out << "\n";
    //activation functions
    out << "activations: ";
    for (string act: activationPerLayer)
        out << act << " ";
    out << "\n";

    //weights
    out << "w: ";
    for (size_t i = 0; i < layerTotal-1; i++){
        for (size_t j = 0; j < nodePerLayer.at(i+1); j++){
            for (size_t k = 0; k < nodePerLayer.at(i); k++){
               out << weights[i][j][k] << " ";
            }
        }
    }
    out << "\n";

    //biases
    out << "b: ";
    for (size_t i = 0; i < layerTotal-1; i++){
        out << biases[i] << " ";
    }
    out << endl;
    
    out.close(); 
}

/**
 * Loads Model information from a .txt file
 * @param filename name of txt file
 * @return None
 */
void DenseNeuralNetwork::load(string filename){ 
    std::ifstream file(filename);
    string formatERR =  "ERROR: File should be in the following format:\n loss: <loss function>\nlayers: <space separated integers>\nactivations: <space separated activation function names>\nw: <space separated doubles>\nb: <space seperated doubles>";

    if (!file.is_open())
        throw "ERROR: File could not be opened";

    string tag, lossNew;
    file >> tag >> lossNew;
    if (tag != "loss:")
        throw formatERR;
    
    string line;
    std::getline(file,line);
    
    //get layers
    line = "";
    vector<size_t> layersNew = {};
    std::getline(file,line);

    string elem = "";
    for (char a: line){
        if (a == ' '){
            if (elem != "layers:") layersNew.push_back(std::stoi(elem));
            elem = "";
            continue;
        }
        elem += a;
    }

    //activations
    line = "";
    vector<string> activeNew = {};
    std::getline(file,line);

    elem = "";
    for (char a: line){
        if (a == ' '){
            if (elem != "activations:") activeNew.push_back(elem);
            elem = "";
            continue;
        }
        elem += a;
    }

    vector<double> w = {};
    vector<double> b = {};

    //weights
    line = "";
    std::getline(file,line);

    elem = "";
    for (char a: line){
        if (a == ' '){
            if (elem != "w:") w.push_back(std::stod(elem));
            elem = "";
            continue;
        }
        elem += a;
    }

    //bias
    line = "";
    std::getline(file,line);

    elem = "";
    for (char a: line){
        if (a == ' '){
            if (elem != "b:") b.push_back(std::stod(elem));
            elem = "";
            continue;
        }
        elem += a;
    }

    //load
    copy(DenseNeuralNetwork(lossNew,layersNew,activeNew)); //fix later
    loadWB(w,b);   
}



/**
 * Updates each of the weights in the DNN
 * @param learning_rate determines magnitude of weight change
 * @param expected 1D vector that contains the expected values in the output node
 * @updates weights
 * @return None
 * 
 * MATH: 
 * w_new = w_old - learning rate * (D_w E), where D_w E is the partial derivative of the Loss Function (E) with respect to w
 * 
 * D_w E = oi * deltaj, where oi is the previous node's value
 * 
 * deltaj = D_oj E * D_netj oj, where oj is the current node's value (the current weight w is the weight from oi to oj), 
 * and netj is the summation for the previous layers nodes multiplied by each of their corresponding weights pointing to oj
 * 
 ** D_netj oj = The partial deriviative of the activation function on the current layer (where node oj is the input) 
 * ex: If the Activation Function is Logistic, D_netj oj = oj*(1 - oj)
 * 
 * D_oj E has a recursive definition.
 * If oj is in an output layer,
 * D_oj E = The partial derivative of the loss function (which accepts oj as a node and tj, the target value for oj) wrt oj 
 * ex: If the Loss Function is Mean Squared Error, D_oj E = oj - tj
 * 
 * If oj is in an inner layer,
 * D_oj E = Sum of (weights from oj to ol in the next layer multiplied by deltal) for all ol in nodeL (if nodeL is a set of all nodes in the next layer, ol is an element of that set)
 * 
 * IMPLEMENTATION:
 * On any given iteration of the update function,
 * 
 * weights[i][j][k] = the weight from node[i][k] to node[i+1][j]
 * 
 * If nodes[i][j] is output, 
 * delta[i-1][j] = D_nodes[i][j] L(nodes[i][j], expected.at(j)) * D of activation function with input (node[i][j]), where L is the Loss function that accepts a node and its target value IF output node
 * 
 * Else, 
 * For nextNode in 0 to amount of nodes in next layer:
 *      delta[i-1][j] += (weights[i][nextNode][j] * delta[i][nextNode]) 
 * delta[i-1][j] *=  D of activation function with input (nodes[i][j])
 * 
 * weights[i][j][k] -= learning_rate * nodes[i][k] * delta[i][j]
 */
void DenseNeuralNetwork::update(double learning_rate, vector<double> expected){

    size_t i = layerTotal-1;
    //Base
    for (size_t j = 0; j < nodePerLayer.at(i); j++){ //compare each of the output nodes to their expected values * derivative of activation function                
        //Derivative of Loss Function
        delta[i-1][j] = D_L(nodes[i][j],expected.at(j));

        //Derivative of Activation Function
        delta[i-1][j] *= activationDerivatives.at(i-1)(nodes[i][j]);

        //just update weights here
        for (size_t k = 0; k < nodePerLayer.at(i-1); k++){
            weights[i-1][j][k] -= (learning_rate* nodes[i-1][k]*delta[i-1][j]);
        }
        biases[i-1] -= learning_rate * delta[i-1][j];// bias update is b[i] -= lr * delta[i]
    }
    i--;

    //Recursive
    for (; i > 0; i--){        
        for (size_t j = 0; j < nodePerLayer.at(i); j++){  
            delta[i-1][j] = 0;

            //calculate the summation of all of the weights from Node j to each of the nodes in the next layer * S of each of those nodes in the next layer ^ derivative of activation func
            for (size_t nextNode = 0; nextNode < nodePerLayer.at(i+1); nextNode++ ){
                delta[i-1][j] += (weights[i][nextNode][j] * delta[i][nextNode]); //weights[connected section index][number of node in next layer][number of node in prev layer]
            }
            
            //Derivative of Activation Function
            delta[i-1][j] *= activationDerivatives.at(i-1)(nodes[i][j]);
            
            //just update weights here
            for (size_t k = 0; k < nodePerLayer.at(i-1); k++){
               weights[i-1][j][k] -= (learning_rate* nodes[i-1][k]*delta[i-1][j]);
            }
            
            biases[i-1] -= learning_rate * delta[i-1][j];// bias update is b[i] -= lr * delta[i]   
        }
    }
}



/**
 * Computes the values of a single layer of the DNN based on the current weights and biases and on the values of the previous layers nodes
 * @param layer represents which layer is being computed (will use the previous layer (layer-1() in computation)
 * @updates nodes[layer]
 * @return None
 */
void DenseNeuralNetwork::computeLayer(size_t layer){ //sofar good
    if (layer >= layerTotal)
        throw std::out_of_range("ERROR: Attempted to compute the value of the input layer or a non-exsitent layer");
    
    //find activation function
    auto activate = activation.at(layer-1);
    
    for (size_t j = 0; j < nodePerLayer.at(layer); j++){
        double sum = 0;

        //computation formula is node[layer][j] = Activation(Sigma prev=0 to n (node[layer-1][prev] * weight from prev to j) + bias)
        for (size_t prev = 0; prev < nodePerLayer.at(layer-1); prev++){
            sum += nodes[layer-1][prev] * weights[layer-1][j][prev]; 
        }
        nodes[layer][j] = activate(sum + biases[layer-1]);
    }

    if (activationPerLayer.at(layer-1) == "softmax"){
        softmax(nodes[layer], nodePerLayer.at(layer));
    } 
}

/**
 * Trains Neural Network by updating weights based on user given labels and features
 * @param data training input feature data
 * @param expected training input label data
 * @param activations vector of strings that contain the same of the activation function used on the corresponding inner/outer layers of the DNN
 * @param epochs number of times the DNN will view the same dataset
 * @param learning_rate determines magnitude of weight change
 * @param isAccurate function that compares vector of predicted output and actual output, returns boolean if predicted output correctly predicts actual output
 * @param verbose boolean to depict current progress of training
 * @param shuffle boolean that decides if dataset should be shuffled
 * @updates weights, nodes
 * @return None
 */
void DenseNeuralNetwork::train(vector<vector<double>> data, vector<vector<double>> expected, size_t epochs, double learning_rate, bool (*isAccurate)(vector<double>, vector<double>) = nullptr, bool verbose = true, bool shuffle = false){
    if (data.size() != expected.size())
        throw "ERROR: Dimensions of training dataset does not match number of labels";

    if(shuffle){
        const unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

        std::shuffle(data.begin(), data.end(), std::default_random_engine(seed)); //shuffles the same
        std::shuffle(expected.begin(), expected.end(), std::default_random_engine(seed));
    }
    
    for (size_t iterations = 0; iterations < epochs; iterations++){
        int acc = 0;
        double lossNum = 0;
        if (verbose){
            cout << "EPOCH " << iterations+1 << ": Training... " <<endl;
        }
        for (size_t i = 0; i < data.size(); i++){
            if (data.at(i).size() != nodePerLayer.at(0))
                throw "ERROR: Dimensions of training dataset does not match number of input nodes";
            vector<double> pred = predict(data.at(i));

            update(learning_rate, expected.at(i));

            //only makes sense if one output layer and using cross entropy function
            if (verbose){
                for (size_t j = 0; j < pred.size(); j++){
                    //lossNum += -1*expected.at(i).at(j)*log2(pred.at(j));
                    lossNum += L(pred.at(j), expected.at(i).at(j));
                }
                if (isAccurate(pred,expected.at(i)))
                    acc++;
            }     
        }
        if (verbose){
            cout << "DONE. Accuracy: " << (double) acc / data.size() << " Loss: " << lossNum / data.size() << endl;
        }
    }
}

/**
 * Tests Neural Network accuracy via testing dataset. Prints accuracy percentage and a confidence interval IF the central limit theorem can be applied
 * @param data testing input feature data
 * @param expected testing input label data
 * @param isAccurate function that compares vector of predicted output and actual output, returns boolean if predicted output correctly predicts actual output
 * @updates nodes
 * @return None
 */
void DenseNeuralNetwork::test(vector<vector<double>> data, vector<vector<double>> expected, bool (*isAccurate)(vector<double>, vector<double>) = nullptr){
    int acc = 0;
    double lossNum = 0;

    for (size_t i = 0; i < data.size(); i++){
        if (data.at(i).size() != nodePerLayer.at(0))
            throw "ERROR: Dimensions of training dataset does not match number of input nodes";
        
        vector<double> pred = predict(data.at(i));
        
        //only makes sense if one output layer and using cross entropy function
        for (size_t j = 0; j < pred.size(); j++){
            lossNum += L(pred.at(j), expected.at(i).at(j));
        }
        if (isAccurate(pred,expected.at(i)))
            acc++;    
    }

    long double totalAcc = (double) acc / data.size();

    cout << "Accuracy: " << totalAcc;
    //Using basic statistical assumption that CLT applies to proportions if n*p and n*q >= 5
    if (totalAcc *  data.size() >= 5 &&  (1-totalAcc) *  data.size() >= 5){
        cout << " +/- " << 2.5758*sqrt(totalAcc*(1-totalAcc)/data.size()) << " (99% CI)";
    }
    cout << " Loss: " << lossNum / data.size() << endl;
}


/**
 * Makes a prediction based on current weights and biases of a given input
 * @param input 1D vector of input data 
 * @updates nodes
 * @return vector of doubles of the output layer.
 */
vector<double> DenseNeuralNetwork::predict(vector<double> input){ //sofar good
    if (input.size() != nodePerLayer.at(0)){
        throw "ERROR: Data's size does not match number of nodes in this Model's Input Layer";
    }
    for (size_t i = 0; i < input.size(); i++){
        nodes[0][i] = input[i];
    }

    //compute each layer starting at 1
    for (size_t i = 1; i < layerTotal; i++){
        computeLayer(i);   
    }

    //copy output layer into a vector
    vector<double> result(nodePerLayer.at(nodePerLayer.size()-1));
    for (size_t i = 0; i < result.size(); i++){
        result.at(i) = nodes[layerTotal-1][i];
    }

    return result;
}

