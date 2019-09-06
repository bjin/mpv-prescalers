
CXXFLAGS = -std=c++11 -O2

.PHONY: all clean

all: ravu ravu-zoom ravu-lite ravu-3x

clean:
	rm -f ravu ravu-zoom ravu-lite ravu-3x
