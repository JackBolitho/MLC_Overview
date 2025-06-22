# How To Run threaded_engine_interface.cc

In order to clone the right version of MLC-LLM, run:
` ./clone_mlc_repo.sh `

Then, to ccompile the threaded engine, run:
` ./compile_threaded_engine.sh `

Call the threaded engine from the parent directory of the build directory like so:
` ./build/threaded_engine `

The threaded_engine program will only run properly if the weights are provided in the parent directory.
