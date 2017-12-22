# Projection-Imaging
1. Make sure you have a Nvidia GPU with the latest driver and CUDA toolkit installed (CUDA 8.0/7.5 both works for me).
2. Once you set up the CUDA environment, go to the directory where all the files are.
3. Specify the h, H, offset_x, offset_y and D in main.cpp.
4. Run the makefile to produce an executable file and then run it, result is stored in ‘Projection.txt’.
5. To plot, just import the .txt file into MATLAB as NUMERIC MATRIX and run plot.m. Remember to change the parameter of axis accordingly to produce a good figure. 
6. Go back to step 3 if you want to do another simulation.
