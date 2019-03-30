#!/usr/bin/env python3

import sys
import os

def main():
	if len(sys.argv) < 2:
		print("Please give an example file name")
		return 0
	name = sys.argv[1]
	modulename = 'examples.' + name

	example = __import__(modulename)
	print(modulename, example)
	example.__getattribute__(name).main()

if __name__ == '__main__':
	main()