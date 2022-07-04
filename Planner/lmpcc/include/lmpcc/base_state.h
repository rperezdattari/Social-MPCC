//
// Created by bdebrito on 16-2-21.
//

#ifndef BASE_STATE_H
#define BASE_STATE_H

class BaseState{

public:
    BaseState()
    {

    }

protected:

    // All possible values that can be in the state
    double x;
    double y;
    double psi;
    double v;
    double delta;
    double ax;
    double ay;
    double w;
    double j;
    double alpha;

public:

    // Default set functions do nothing when set
    // These are overwritten in the actual models to set the variables if they are part of the model
    // I.e., if the variable does not exist, nothing happens
    virtual void set_x(double value){};
    virtual void set_y(double value){};
    virtual void set_psi(double value){};
    virtual void set_v(double value){};
    virtual void set_delta(double value){};
    virtual void set_ax(double value){};
    virtual void set_ay(double value){};
    virtual void set_w(double value){};
    virtual void set_j(double value){};
    virtual void set_alpha(double value){};

    // Default get functions return an error
    // These are overwritten in the actual models to get the variables if they are part of the model
    // I.e., if the variable does not exist, an error is thrown
    virtual double &get_x() { throw std::runtime_error("BaseState: State x was accessed, but does not exist"); };
    virtual double &get_y() { throw std::runtime_error("BaseState: State y was accessed, but does not exist"); };
    virtual double &get_psi() { throw std::runtime_error("BaseState: State psi was accessed, but does not exist"); };
    virtual double &get_v() { throw std::runtime_error("BaseState: State v was accessed, but does not exist"); };
    virtual double &get_delta() { throw std::runtime_error("BaseState: State delta was accessed, but does not exist"); };
    virtual double &get_ax() { throw std::runtime_error("BaseState: State ax was accessed, but does not exist"); };
    virtual double &get_ay() { throw std::runtime_error("BaseState: State ay was accessed, but does not exist"); };
    virtual double &get_w() { throw std::runtime_error("BaseState: State w was accessed, but does not exist"); };
    virtual double &get_j() { throw std::runtime_error("BaseState: State j was accessed, but does not exist"); };
    virtual double &get_alpha() { throw std::runtime_error("BaseState: State alpha was accessed, but does not exist"); };

    virtual 
    void init() = 0;

    virtual
    void print() = 0;

};

#endif //BASE_STATE_H
