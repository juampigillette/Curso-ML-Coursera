
import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0, 2.0])   #features
y_train = np.array([300.0, 500.0])   #target value

def funcion_lineal(x_modelo,pendiente_m,odo_b):
  return pendiente_m *x_modelo+ odo_b

def funcion_Costos(x, y, w, b):
   
    m = x.shape[0] 
    cost = 0
    
    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2
    total_cost = 1 / (2 * m) * cost

    return total_cost

def gradient_descent(x, y, w_inicial, b_inicial, alpha, num_iters, cost_function, gradient_function): 
    """
    Performs gradient descent to fit w,b. Updates w,b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x (ndarray (m,))  : Data, m examples 
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): valores iniciales w y b arbitrarios
      alpha (float):     Learning rate
      num_iters (int):   numero de iteraciones para actualizar los parametros
      cost_function:     funcion de costos
      gradient_function: funcion a llamar que produce el los gradientes
      
    Returns:
      w (scalar): valor w actualizado, post algoritmo de gradiente descendiente
      b (scalar): valor b actualizado, post algoritmo de gradiente descendiente
      J_history (List): Historia de los valores de costo J(w,b)
      p_history (list): Historia de los parametros w y b
      """
    
    J_history = [] 
    p_history = []
    b = b_inicial
    w = w_inicial
  
    for i in range(num_iters):
        # Calculamos el gradiente para poder utilizarlo en la actualizacion de los parametros
        dj_dw, dj_db = gradient_function(x, y, w , b)     

        # Actualizo parametros
        b = b - alpha * dj_db                            
        w = w - alpha * dj_dw                            

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(x, y, w , b))
            p_history.append([w,b])

    return w, b, J_history, p_history #return w and J,w history for graphing
