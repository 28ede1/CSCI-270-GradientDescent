import torch
from levels import Environment


def vanilla_grad_descent(rate, env):
    """Basic gradient descent.
    
    This is completed for you, but it won't work until you
    correctly implement a subroutine called grad_descent
    (a stub implementation is found below, fill in the code).
    
    """
    def vanilla_step_fn(pos):
        return -rate * env.gradient(pos)
    return grad_descent(vanilla_step_fn, env)


def grad_descent(step_fn, env):
    """
    A general-purpose gradient descent algorithm.
    
    step_fn is a function that takes a position (x,y) as input 
    (expressed as a 2-element torch.tensor), and returns the
    relative step to take (also expressed as a 2-element torch.tensor).
    
    env is the environment.
    
    The return value should be a list of the positions (including
    the starting position) visited during the gradient descent. 
    
    """
    
    initial_position = env.current_position()
    visited_positions = [initial_position]

    while env.status() == Environment.ACTIVELY_SEARCHING:
        current_position = env.current_position()
        delta_position = step_fn(current_position)
        new_position = current_position + delta_position
        env.step_to(new_position)
        visited_positions.append(new_position)

    return visited_positions
    


def momentum_grad_descent(rate, env):
    """Gradient descent with momentum.
    
    This is completed for you, but it won't work until you
    correctly implement the MomentumStepFunction class.
    (a stub implementation is found below, fill in the code).
    
    """
    return grad_descent(MomentumStepFunction(env.gradient, rate, 0.3), env)


class MomentumStepFunction:
    """
    Computes the next step for gradient descent with momentum.

    The __call__ method takes a position (x,y) as its argument (expressed
    as a 2-dimensional torch.tensor), and returns the next relative step
    that gradient descent with momentum would take (also expressed as a
    2-dimensional torch.tensor).
        
    """    
    def __init__(self, loss_gradient, learning_rate, momentum_rate):
        # Question TWO
        self.loss_function = loss_gradient
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.previous_pos = None

    def __call__(self, pos):
        # Question TWO

        if self.previous_pos is None:
            momentum = torch.tensor([0.0, 0.0])
        else:
            momentum = self.momentum_rate * (pos - self.previous_pos)
        next_step = -self.learning_rate * self.loss_function(pos) + momentum
        self.previous_pos = pos
        return next_step



def adagrad(rate, env):
    """Adaptive gradient descent (adagrad).
    
    This is completed for you, but it won't work until you
    correctly implement the AdagradStepFunction class.
    (a stub implementation is found below, fill in the code).
    
    """
    return grad_descent(AdagradStepFunction(env.gradient, rate), env)


class AdagradStepFunction:
    """
    Computes the next step for adagrad.

    The __call__ method takes a position (x,y) as its argument (expressed
    as a 2-dimensional torch.tensor), and returns the next relative step
    that adagrad would take (also expressed as a
    2-dimensional torch.tensor).
        
    """
    def __init__(self, loss_gradient, learning_rate, delta = 0.0000001):
        # Question THREE
        self.loss_function = loss_gradient
        self.learning_rate = learning_rate
        self.delta = delta
        self.previous_pos = None

        # keeps a running sum of the squared gradient values
        self.odemeter_slope_track = torch.tensor([0.0, 0.0])
        
    def __call__(self, pos):
        # Question THREE

        # calculate gradient at a position and then suqare. add this to the total. then square root that total

        current_gradient_pos_value = self.loss_function(pos)
        self.odemeter_slope_track += (current_gradient_pos_value ** 2)
        new_learning_rate = (self.learning_rate) / (self.delta + torch.sqrt(self.odemeter_slope_track))  
        return -new_learning_rate * current_gradient_pos_value  


def rmsprop(rate, decay_rate, env):
    """The RMSProp variant of gradient descent.
    
    This is completed for you, but it won't work until you
    correctly implement the RmsPropStepFunction class.
    (a stub implementation is found below, fill in the code).
    
    """
    return grad_descent(RmsPropStepFunction(env.gradient, rate, decay_rate), env)


class RmsPropStepFunction:
    """
    Computes the next step for RmsProp.

    The __call__ method takes a position (x,y) as its argument (expressed
    as a 2-dimensional torch.tensor), and returns the next relative step
    that RmsProp would take (also expressed as a
    2-dimensional torch.tensor).
        
    """
    def __init__(self, loss_gradient, learning_rate, decay_rate, delta=0.000001):
        # Question FOUR
        pass
        
    def __call__(self, pos):
        # Question FOUR
        pass
