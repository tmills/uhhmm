#ifndef obs_models_hpp
#define obs_models_hpp

#include <stdio.h>
#include "myarrays.h"
#include "HmmSampler.h"
//#include "Indexer.h"

class Model;
class Indexer;

class ObservationModel{
protected:
    Indexer * p_indexer = NULL;
public:
    virtual ~ObservationModel(){}
    virtual void set_models(Model * models) = 0;
    virtual void get_probability_vector(int token, Array* output) = 0;
};

class PosDependentObservationModel : public ObservationModel {
private:
    Sparse* lexMultiplier = NULL;
    int g;
public:
    virtual ~PosDependentObservationModel();
    virtual void set_models(Model * models);
    virtual void get_probability_vector(int token, Array* output);
    virtual void get_pos_probability_vector(int token, Array* output) = 0;
};

class GaussianObservationModel : public PosDependentObservationModel {
private:
    DenseView* lexMatrix = NULL;
    DenseView* embeddings = NULL;
    int embed_dims;
    thrust::device_vector<float> *temp = NULL;
    //DenseView*** distributions = NULL;
public:
    ~GaussianObservationModel(){
        PosDependentObservationModel::~PosDependentObservationModel();
        delete lexMatrix;
    }
    virtual void set_models(Model * models);
    virtual void get_pos_probability_vector(int token, Array* output);
};

class CategoricalObservationModel : public PosDependentObservationModel {
private:
    DenseView* lexMatrix = NULL;
public:
    ~CategoricalObservationModel();
    virtual void set_models(Model * models);
    virtual void get_pos_probability_vector(int token, Array* output);
};

#endif
