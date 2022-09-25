# You have to implement Karplus Strong alogrithm, by the formula 
# y[n] = x[n] + a * y[n-M], where 0 < a < 1, x[n] is finite support, x[n] = y[n] = 0 if n < 0.
# 1) you have to create function which takes vector x, a, and N as length of output signal 
# and returns y[n]
# 2) you have to create function which generates random x vector, feeds it to previous function
# and saves y[n] output signal as .wav file
# 3) Using last function you have to create wav examples
