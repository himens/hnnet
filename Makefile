# Exec
EXE := test.exe

# Compiler
CXX := g++

# Paths
SRC := src
INC := include
OBJ := obj

# Compiler flags
CXXFLAGS := -Wall -fPIC -O3 -std=c++11 # C++11

# Src, obj, inc
SRCS := $(wildcard $(SRC)/*.cpp)
OBJS := $(patsubst $(SRC)/%.cpp, $(OBJ)/%.o, $(SRCS))

.PHONY: all clean

all: $(EXE)

# Make exec
$(EXE): $(OBJS) 
	$(CXX) $(CXXFLAGS) $^ -o $@ 

# Make objs
$(OBJ)/%.o: $(SRC)/%.cpp $(INC)
	@mkdir -p $(OBJ)
	$(CXX) $(CXXFLAGS) -c $< -o $@ -I$(INC)

# Clean
clean:
	-rm -r $(OBJ)
	-rm $(EXE)
