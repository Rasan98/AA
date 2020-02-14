#Se importan las librerías que se van a necesitar
import matplotlib.pyplot as plt
import time
import numpy as np
import random

#Definimos una variable global para llevar el número de iteraciones (para imprimir el resultado):
iteracion = 1

#Definimos nuestra función a integrar:
def function(x):
    return 2*x


#Función que utiliza bucles para obtener el resultado de la integral
def integra_mc_listas(fun, a ,b, num_puntos = 10000):
   

    start = time.process_time()
    
    axis_x = [random.uniform(a, b) for n in range(num_puntos)]
    aux = [fun(n) for n in axis_x]
    
    max_fun = max(aux)
    
    maxs = [random.uniform(0, max_fun) for n in range(num_puntos)]
    
    booleans = [ maxs[i] <= aux[i] for i in range(0,num_puntos)] 
    
    points = booleans.count(True)

    print("Iteracion ", iteracion, " (listas) -> ", (points / num_puntos)*(b - a)* max_fun)

    stop = time.process_time()

    return 1000 * (stop - start)

#Función que utiliza operaciones vectoriales para obtener el resultado de la integral
def integra_mc_vectorial(fun, a ,b, num_puntos = 10000):
    
    start = time.process_time()
    
    axis_x = np.random.uniform(a, b, num_puntos)

    aux = np.apply_along_axis(fun, 0, axis_x)
    
    max_fun = aux.max()
    
    maxs = np.random.uniform(0, max_fun, num_puntos)

    booleans = maxs <= aux 
    
    points = np.sum(booleans)

    print("Iteracion ", iteracion, " (vectorial) -> ", (points / num_puntos) * (b - a) * max_fun)
        
    stop = time.process_time()

    return 1000 * (stop - start)

#Función que crea la figura pedida en el enunciado 
def compara_tiempos(fun):
    global iteracion
    sizes = np.linspace(100,10000000,20)
    sizes.sort() 
    times_vector=[]
    times_list=[]
    for size in sizes:
        times_vector += [integra_mc_vectorial(fun, 1, 4, num_puntos=int(size))]
        times_list += [integra_mc_listas(fun, 1, 4, num_puntos=int(size))]
        iteracion += 1
    
    plt.figure()
    plt.scatter(sizes, times_vector, c='red', label='Operaciones Vectoriales (Numpy)')
    plt.scatter(sizes, times_list, c='blue', label='Bucles')
    plt.legend()
    plt.savefig('time.png')
    print("Finished")

#Llamada a la funcion para crear la gráfica
compara_tiempos(function)