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
    virtual Array * get_probability_vector(int token) = 0;
};

class PosDependentObservationModel : public ObservationModel {
private:
    Sparse* lexMultiplier = NULL;

public:
    virtual ~PosDependentObservationModel();
    virtual void set_models(Model * models);
    virtual Array * get_probability_vector(int token);
    virtual array2d<float, device_memory>::column_view get_pos_probability_vector(int token) = 0;

};

class GaussianObservationModel : public PosDependentObservationModel {
private:
    DenseView* lexMatrix = NULL;
public:
    ~GaussianObservationModel(){
        PosDependentObservationModel::~PosDependentObservationModel();
        delete lexMatrix;
    }
    virtual void set_models(Model * models);
    virtual array2d<float, device_memory>::column_view get_pos_probability_vector(int token);
};

class CategoricalObservationModel : public PosDependentObservationModel {
private:
    DenseView* lexMatrix = NULL;
public:
    ~CategoricalObservationModel();
    virtual void set_models(Model * models);
    virtual array2d<float, device_memory>::column_view get_pos_probability_vector(int token);
};

#endif
