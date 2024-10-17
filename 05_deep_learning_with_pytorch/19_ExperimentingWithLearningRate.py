#In this exercise, your goal is to find the optimal learning rate such that the optimizer can find the minimum of the non-convex function in ten steps.
#You will experiment with three different learning rate values. For this problem, try learning rate values between 0.001 to 0.1.
#You are provided with the optimize_and_plot() function that takes the learning rate for the first argument. This function will run 10 steps of the SGD optimizer and display the results.
#
# Instructions 1/3
#
#Try a small learning rate value such that the optimizer isn't able to get past the first minimum on the right.
#
#Code
#
## Try a first learning rate value
#lr0 = 0.084
#optimize_and_plot(lr=lr0)
#
# Instructions 2/3
#
# Try a large learning rate value such that the optimizer skips past the global minimum at -2.
#
#Code
#
## Try a second learning rate value
#lr1 = 0.1
#optimize_and_plot(lr=lr1)
#
# Instruction 3/3
#
#Based on the previous results, try a better learning rate value.
#
#Code
#
## Try a third learning rate value
#lr2 = 0.084
#optimize_and_plot(lr=lr2)
